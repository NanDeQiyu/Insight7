// bindings/python/insight_py.cpp
//
// Python bindings for the Insight7 scientific computing framework.
// Built with pybind11, wrapping the C++ ins:: namespace API.
// API style: PaddlePaddle (ins.float32, ins.CPUPlace(), etc.)

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "insight/c_api/array.h"
#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/slice.h"
#include "insight/init.h"
#include "insight/ops/cast.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/random.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
#endif

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace ins;

// ============================================================================
// DType / Place helpers
// ============================================================================

static const char *dtype_to_str(DType dt) {
  switch (dt) {
  case DType::BOOL:
    return "bool";
  case DType::U8:
    return "uint8";
  case DType::I8:
    return "int8";
  case DType::I16:
    return "int16";
  case DType::I32:
    return "int32";
  case DType::I64:
    return "int64";
  case DType::U16:
    return "uint16";
  case DType::U32:
    return "uint32";
  case DType::U64:
    return "uint64";
  case DType::F16:
    return "float16";
  case DType::BF16:
    return "bfloat16";
  case DType::F32:
    return "float32";
  case DType::F64:
    return "float64";
  case DType::C32:
    return "complex64";
  case DType::C64:
    return "complex128";
  default:
    return "unknown";
  }
}

// ============================================================================
// NumPy <-> Insight Array conversion
// ============================================================================

static py::dtype dtype_to_numpy(DType dt) {
  switch (dt) {
  case DType::BOOL:
    return py::dtype::of<bool>();
  case DType::U8:
    return py::dtype::of<uint8_t>();
  case DType::I8:
    return py::dtype::of<int8_t>();
  case DType::I16:
    return py::dtype::of<int16_t>();
  case DType::I32:
    return py::dtype::of<int32_t>();
  case DType::I64:
    return py::dtype::of<int64_t>();
  case DType::U16:
    return py::dtype::of<uint16_t>();
  case DType::U32:
    return py::dtype::of<uint32_t>();
  case DType::U64:
    return py::dtype::of<uint64_t>();
  case DType::F32:
    return py::dtype::of<float>();
  case DType::F64:
    return py::dtype::of<double>();
  case DType::C32:
    return py::dtype::of<std::complex<float>>();
  case DType::C64:
    return py::dtype::of<std::complex<double>>();
  default:
    throw std::invalid_argument("Unsupported dtype for NumPy conversion");
  }
}

static DType numpy_dtype_to_insight(const py::dtype &dt) {
  if (dt.is(py::dtype::of<bool>()))
    return DType::BOOL;
  if (dt.is(py::dtype::of<uint8_t>()))
    return DType::U8;
  if (dt.is(py::dtype::of<int8_t>()))
    return DType::I8;
  if (dt.is(py::dtype::of<int16_t>()))
    return DType::I16;
  if (dt.is(py::dtype::of<int32_t>()))
    return DType::I32;
  if (dt.is(py::dtype::of<int64_t>()))
    return DType::I64;
  if (dt.is(py::dtype::of<uint16_t>()))
    return DType::U16;
  if (dt.is(py::dtype::of<uint32_t>()))
    return DType::U32;
  if (dt.is(py::dtype::of<uint64_t>()))
    return DType::U64;
  if (dt.is(py::dtype::of<float>()))
    return DType::F32;
  if (dt.is(py::dtype::of<double>()))
    return DType::F64;
  if (dt.is(py::dtype::of<std::complex<float>>()))
    return DType::C32;
  if (dt.is(py::dtype::of<std::complex<double>>()))
    return DType::C64;
  throw std::invalid_argument("Unsupported NumPy dtype");
}

static Array from_numpy(py::array arr) {
  arr = arr.attr("astype")(dtype_to_numpy(numpy_dtype_to_insight(arr.dtype())))
            .attr("copy")();
  auto info = arr.request();
  DType dt = numpy_dtype_to_insight(arr.dtype());

  std::vector<int64_t> shape(info.shape.begin(), info.shape.end());
  Array result(Shape(shape), dt, CPUPlace());
  std::memcpy(result.data(), info.ptr, arr.nbytes());
  return result;
}

// Recursively infer shape from a nested Python list
static void infer_list_shape(py::list lst, std::vector<int64_t> &shape,
                             int depth) {
  if (depth >= INSIGHT_MAX_NDIM) {
    throw std::invalid_argument(
        "List nesting exceeds maximum supported dimensions (" +
        std::to_string(INSIGHT_MAX_NDIM) + ")");
  }
  int64_t len = static_cast<int64_t>(lst.size());
  if (depth >= static_cast<int>(shape.size())) {
    shape.push_back(len);
  } else if (shape[depth] != len) {
    throw std::invalid_argument(
        "Ragged nested list: dimension " + std::to_string(depth) +
        " has inconsistent sizes (" + std::to_string(shape[depth]) + " vs " +
        std::to_string(len) + ")");
  }
  if (len > 0 && py::isinstance<py::list>(lst[0])) {
    for (int64_t i = 0; i < len; i++) {
      if (!py::isinstance<py::list>(lst[i])) {
        throw std::invalid_argument("Mixed nesting at dimension " +
                                    std::to_string(depth) + ": element " +
                                    std::to_string(i) +
                                    " is not a list but siblings are");
      }
      infer_list_shape(py::reinterpret_borrow<py::list>(lst[i]), shape,
                       depth + 1);
    }
  }
}

// Flatten a nested Python list into a vector of doubles
static void flatten_list(py::list lst, std::vector<double> &out) {
  for (size_t i = 0; i < lst.size(); i++) {
    py::object item = lst[i];
    if (py::isinstance<py::list>(item)) {
      flatten_list(py::reinterpret_borrow<py::list>(item), out);
    } else {
      out.push_back(item.cast<double>());
    }
  }
}

static Array from_array_impl(py::object obj) {
  // If already a numpy array, use from_numpy
  if (py::isinstance<py::array>(obj)) {
    return from_numpy(py::reinterpret_borrow<py::array>(obj));
  }
  // If a Python list, convert via shape inference + flatten
  if (py::isinstance<py::list>(obj)) {
    py::list lst = py::reinterpret_borrow<py::list>(obj);
    if (lst.size() == 0) {
      return Array(Shape({0}), DType::F64, CPUPlace());
    }
    std::vector<int64_t> shape;
    infer_list_shape(lst, shape, 0);
    std::vector<double> data;
    flatten_list(lst, data);
    int64_t expected = 1;
    for (auto d : shape)
      expected *= d;
    if (static_cast<int64_t>(data.size()) != expected) {
      throw std::invalid_argument("Shape/data size mismatch");
    }
    Array result(Shape(shape), DType::F64, CPUPlace());
    std::memcpy(result.data(), data.data(), data.size() * sizeof(double));
    return result;
  }
  throw std::invalid_argument("from_array expects a list or numpy array, got " +
                              std::string(py::str(obj.get_type())));
}

static py::array to_numpy(const Array &arr) {
  Array cpu = arr.contiguous().to(CPUPlace());
  DType dt = cpu.dtype();
  py::dtype np_dt = dtype_to_numpy(dt);

  std::vector<ssize_t> shape;
  std::vector<ssize_t> strides;
  for (int i = 0; i < cpu.shape().ndim(); i++) {
    shape.push_back(cpu.shape().dim(i));
    strides.push_back(cpu.strides()[i] * dtype_size(dt));
  }

  return py::array(np_dt, shape, strides, cpu.data());
}

// ============================================================================
// Array __repr__
// ============================================================================

static std::string array_repr(const Array &a) {
  if (!a.defined())
    return "<insight.Array (undefined)>";
  char *s = insight_array_tostring(a.layout_ptr());
  if (s) {
    std::string result(s);
    std::free(s);
    return result;
  }
  return "<insight.Array (error)>";
}

// ============================================================================
// Module definition
// ============================================================================

PYBIND11_MODULE(_insight, m) {
  m.doc() = "Insight7 — lightweight scientific computing framework";
  m.attr("__version__") = "1.0.0";

  // Auto-initialize CPU backend on module load
  if (!ins::is_initialized()) {
    ins::init({"cpu"});
  }

  // Manual init for advanced users
  m.def(
      "init",
      [](const std::vector<std::string> &backends) { ins::init(backends); },
      py::arg("backends") = std::vector<std::string>{"cpu"},
      "Initialize Insight backends");
  m.def("is_initialized", &ins::is_initialized,
        "Check if Insight is initialized");
  m.def(
      "load_backend",
      [](const std::string &backend) {
        try {
          ins::load_backend(backend);
        } catch (const std::exception &e) {
          throw py::value_error(e.what());
        }
      },
      py::arg("backend"),
      "Load an additional backend after init() (e.g., 'cuda', 'rocm')");

  // ===== Device information =====
  m.def(
      "device_name",
      [](const std::string &kind, int device_id) {
        DeviceKind dk = (kind == "gpu" || kind == "cuda") ? DeviceKind::GPU
                                                          : DeviceKind::CPU;
        return device_name(dk, device_id);
      },
      py::arg("kind") = "cpu", py::arg("device_id") = 0,
      "Get the name of a device");
  m.def(
      "cuda_version", []() { return cuda_version(); },
      "Get the CUDA runtime version (major*1000+minor*10, 0 if not available)");
  m.def(
      "driver_version", []() { return driver_version(); },
      "Get the CUDA driver version (major*1000+minor*10, 0 if not available)");
  m.def(
      "compute_capability",
      [](int device_id) { return compute_capability(device_id); },
      py::arg("device_id") = 0,
      "Get the compute capability of a GPU (e.g., 80 for SM 8.0)");
  m.def(
      "device_memory",
      [](int device_id) {
        auto info = device_memory(device_id);
        return py::make_tuple(info.total, info.free);
      },
      py::arg("device_id") = 0,
      "Get GPU memory info as (total_bytes, free_bytes)");
  m.def(
      "device_count",
      []() { return static_cast<int>(device_count(DeviceKind::GPU)); },
      "Get the number of GPU devices");

  // ===== DType (Paddle-style: ins.float32, ins.int64, ...) =====
  py::enum_<DType>(m, "DType")
      .value("bool", DType::BOOL)
      .value("uint8", DType::U8)
      .value("int8", DType::I8)
      .value("int16", DType::I16)
      .value("int32", DType::I32)
      .value("int64", DType::I64)
      .value("uint16", DType::U16)
      .value("uint32", DType::U32)
      .value("uint64", DType::U64)
      .value("float16", DType::F16)
      .value("bfloat16", DType::BF16)
      .value("float32", DType::F32)
      .value("float64", DType::F64)
      .value("complex64", DType::C32)
      .value("complex128", DType::C64)
      .export_values();

  m.attr("bool") = DType::BOOL;
  m.attr("uint8") = DType::U8;
  m.attr("int8") = DType::I8;
  m.attr("int16") = DType::I16;
  m.attr("int32") = DType::I32;
  m.attr("int64") = DType::I64;
  m.attr("uint16") = DType::U16;
  m.attr("uint32") = DType::U32;
  m.attr("uint64") = DType::U64;
  m.attr("float16") = DType::F16;
  m.attr("bfloat16") = DType::BF16;
  m.attr("float32") = DType::F32;
  m.attr("float64") = DType::F64;
  m.attr("complex64") = DType::C32;
  m.attr("complex128") = DType::C64;

  // ===== Place (Paddle-style: ins.CPUPlace(), ins.GPUPlace(0)) =====
  py::class_<Place>(m, "Place")
      .def("is_cpu", &Place::is_cpu)
      .def("is_gpu", &Place::is_gpu)
      .def("device_id", &Place::device_id)
      .def("__repr__", [](const Place &p) {
        return p.is_gpu() ? "GPUPlace(" + std::to_string(p.device_id()) + ")"
                          : "CPUPlace()";
      });

  m.def("CPUPlace", []() { return CPUPlace(); }, "Create a CPU place");
  m.def(
      "GPUPlace", [](int id) { return GPUPlace(id); }, py::arg("id") = 0,
      "Create a GPU place");

  // ===== Shape =====
  py::class_<Shape>(m, "Shape")
      .def(py::init<>())
      .def(py::init<std::vector<int64_t>>())
      .def("ndim", &Shape::ndim)
      .def("numel", &Shape::numel)
      .def("dim", &Shape::dim, py::arg("i"))
      .def("__getitem__", [](const Shape &s, int i) { return s.dim(i); })
      .def("__len__", &Shape::ndim)
      .def("__repr__", [](const Shape &s) {
        std::ostringstream ss;
        ss << "(";
        for (int i = 0; i < s.ndim(); i++) {
          if (i > 0)
            ss << ", ";
          ss << s.dim(i);
        }
        ss << ")";
        return ss.str();
      });

  // ===== Slice =====
  py::class_<Slice>(m, "Slice")
      .def(py::init<>())
      .def(py::init<int64_t, int64_t>(), py::arg("start"), py::arg("stop"))
      .def(py::init<int64_t, int64_t, int64_t>(), py::arg("start"),
           py::arg("stop"), py::arg("step"))
      .def_readwrite("start", &Slice::start)
      .def_readwrite("stop", &Slice::stop)
      .def_readwrite("step", &Slice::step);

  // ===== Array =====
  py::class_<Array>(m, "Array")
      .def(py::init<>())
      .def(py::init<const Shape &, DType, const Place &>(), py::arg("shape"),
           py::arg("dtype") = DType::F32, py::arg("place") = get_device())
      // --- properties ---
      .def("shape", [](const Array &a) { return a.shape(); })
      .def("dtype", &Array::dtype)
      .def("place", &Array::place)
      .def("numel", &Array::numel)
      .def("nbytes", &Array::nbytes)
      .def("ndim", [](const Array &a) { return a.shape().ndim(); })
      .def("is_contiguous", &Array::is_contiguous)
      .def("defined", &Array::defined)
      // --- view ops ---
      .def("contiguous", &Array::contiguous)
      .def("reshape", &Array::reshape, py::arg("new_shape"))
      .def("transpose", py::overload_cast<>(&Array::transpose, py::const_))
      .def("transpose",
           py::overload_cast<const std::vector<int> &>(&Array::transpose,
                                                       py::const_),
           py::arg("perm"))
      .def(
          "squeeze",
          [](const Array &a, std::optional<int> axis) {
            return a.squeeze(axis);
          },
          py::arg("axis") = std::nullopt)
      .def("unsqueeze", &Array::unsqueeze, py::arg("dim"))
      // --- device/type conversion ---
      .def("to", py::overload_cast<const Place &>(&Array::to, py::const_),
           py::arg("target"))
      .def("to", py::overload_cast<DType>(&Array::to, py::const_),
           py::arg("dtype"))
      .def("to",
           py::overload_cast<const Place &, DType>(&Array::to, py::const_),
           py::arg("target"), py::arg("dtype"))
      .def("copy", &Array::copy)
      // --- slicing ---
      .def("__getitem__",
           [](const Array &a, const std::string &spec) { return a[spec]; })
      .def("__getitem__", [](const Array &a, const Slice &s) { return a[s]; })
      .def("at",
           [](const Array &a, std::vector<int64_t> idx) { return a.at(idx); })
      // --- arithmetic operators ---
      .def("__add__", [](const Array &a, const Array &b) { return a + b; })
      .def("__add__", [](const Array &a, double b) { return a + b; })
      .def("__radd__", [](const Array &a, double b) { return b + a; })
      .def("__sub__", [](const Array &a, const Array &b) { return a - b; })
      .def("__sub__", [](const Array &a, double b) { return a - b; })
      .def("__rsub__", [](const Array &a, double b) { return b - a; })
      .def("__mul__", [](const Array &a, const Array &b) { return a * b; })
      .def("__mul__", [](const Array &a, double b) { return a * b; })
      .def("__rmul__", [](const Array &a, double b) { return b * a; })
      .def("__truediv__", [](const Array &a, const Array &b) { return a / b; })
      .def("__truediv__", [](const Array &a, double b) { return a / b; })
      .def("__rtruediv__", [](const Array &a, double b) { return b / a; })
      .def("__mod__", [](const Array &a, const Array &b) { return a % b; })
      .def("__mod__", [](const Array &a, double b) { return a % b; })
      .def("__neg__", [](const Array &a) { return -a; })
      .def("__pos__", [](const Array &a) { return +a; })
      .def("__invert__", [](const Array &a) { return ~a; })
      // --- comparison operators ---
      .def("__eq__", [](const Array &a, const Array &b) { return equal(a, b); })
      .def("__ne__",
           [](const Array &a, const Array &b) { return not_equal(a, b); })
      .def("__lt__", [](const Array &a, const Array &b) { return less(a, b); })
      .def("__le__",
           [](const Array &a, const Array &b) { return less_equal(a, b); })
      .def("__gt__",
           [](const Array &a, const Array &b) { return greater(a, b); })
      .def("__ge__",
           [](const Array &a, const Array &b) { return greater_equal(a, b); })
      // --- bool ---
      .def("__bool__", [](const Array &a) { return static_cast<bool>(a); })
      // --- NumPy interop ---
      .def(
          "numpy", [](const Array &a) { return to_numpy(a); },
          "Convert to NumPy array (CPU only)")
      .def_static(
          "from_numpy", [](py::array arr) { return from_numpy(arr); },
          "Create Insight Array from NumPy array")
      // --- repr ---
      .def("__repr__", &array_repr)
      .def("__str__", &array_repr);

  // ====================================================================
  // Creation
  // ====================================================================
  m.def("zeros", &zeros, py::arg("shape"), py::arg("dtype") = DType::F32,
        py::arg("place") = get_device());
  m.def("ones", &ones, py::arg("shape"), py::arg("dtype") = DType::F32,
        py::arg("place") = get_device());
  m.def("full", &full, py::arg("shape"), py::arg("fill_value"),
        py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def("eye", &eye, py::arg("n"), py::arg("m") = -1, py::arg("k") = 0,
        py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def(
      "arange",
      [](double end, DType dtype, const Place &place) {
        return arange(end, dtype, place);
      },
      py::arg("end"), py::arg("dtype") = DType::I64,
      py::arg("place") = get_device());
  m.def(
      "arange",
      [](double start, double end, double step, DType dtype,
         const Place &place) { return arange(start, end, step, dtype, place); },
      py::arg("start"), py::arg("end"), py::arg("step") = 1.0,
      py::arg("dtype") = DType::I64, py::arg("place") = get_device());
  m.def("linspace", &linspace, py::arg("start"), py::arg("stop"),
        py::arg("num"), py::arg("dtype") = DType::F32,
        py::arg("place") = get_device());
  m.def("logspace", &logspace, py::arg("start"), py::arg("stop"),
        py::arg("num"), py::arg("base") = 10.0, py::arg("dtype") = DType::F32,
        py::arg("place") = get_device());
  m.def(
      "from_numpy", [](py::array arr) { return from_numpy(arr); },
      py::arg("array"), "Create Insight Array from NumPy array");
  m.def("from_array", &from_array_impl, py::arg("data"),
        "Create Insight Array from a Python list or NumPy array. "
        "Nested lists are supported; ragged or mixed-type lists raise errors.");
  m.def("zeros_like", &zeros_like, py::arg("arr"));
  m.def("ones_like", &ones_like, py::arg("arr"));

  // ====================================================================
  // Elementwise
  // ====================================================================
  m.def(
      "add", [](const Array &a, const Array &b) { return add(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "sub", [](const Array &a, const Array &b) { return sub(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "mul", [](const Array &a, const Array &b) { return mul(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "div", [](const Array &a, const Array &b) { return div(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "pow", [](const Array &a, const Array &b) { return pow(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "mod", [](const Array &a, const Array &b) { return mod(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "maximum", [](const Array &a, const Array &b) { return maximum(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "minimum", [](const Array &a, const Array &b) { return minimum(a, b); },
      py::arg("a"), py::arg("b"));

  // Comparison
  m.def(
      "equal", [](const Array &a, const Array &b) { return equal(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "not_equal",
      [](const Array &a, const Array &b) { return not_equal(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "greater", [](const Array &a, const Array &b) { return greater(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "less", [](const Array &a, const Array &b) { return less(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "greater_equal",
      [](const Array &a, const Array &b) { return greater_equal(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "less_equal",
      [](const Array &a, const Array &b) { return less_equal(a, b); },
      py::arg("a"), py::arg("b"));

  // Logical
  m.def(
      "logical_and",
      [](const Array &a, const Array &b) { return logical_and(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "logical_or",
      [](const Array &a, const Array &b) { return logical_or(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "logical_xor",
      [](const Array &a, const Array &b) { return logical_xor(a, b); },
      py::arg("a"), py::arg("b"));
  m.def(
      "logical_not", [](const Array &x) { return logical_not(x); },
      py::arg("x"));

  // ====================================================================
  // Unary math
  // ====================================================================
  m.def("abs", [](const Array &x) { return ins::abs(x); }, py::arg("x"));
  m.def("negative", [](const Array &x) { return negative(x); }, py::arg("x"));
  m.def("square", [](const Array &x) { return ins::square(x); }, py::arg("x"));
  m.def("sqrt", [](const Array &x) { return sqrt(x); }, py::arg("x"));
  m.def("exp", [](const Array &x) { return exp(x); }, py::arg("x"));
  m.def("log", [](const Array &x) { return log(x); }, py::arg("x"));
  m.def("log2", [](const Array &x) { return log2(x); }, py::arg("x"));
  m.def("log10", [](const Array &x) { return log10(x); }, py::arg("x"));
  m.def("sin", [](const Array &x) { return sin(x); }, py::arg("x"));
  m.def("cos", [](const Array &x) { return cos(x); }, py::arg("x"));
  m.def("tan", [](const Array &x) { return tan(x); }, py::arg("x"));
  m.def("asin", [](const Array &x) { return asin(x); }, py::arg("x"));
  m.def("acos", [](const Array &x) { return acos(x); }, py::arg("x"));
  m.def("atan", [](const Array &x) { return atan(x); }, py::arg("x"));
  m.def("sinh", [](const Array &x) { return sinh(x); }, py::arg("x"));
  m.def("cosh", [](const Array &x) { return cosh(x); }, py::arg("x"));
  m.def("tanh", [](const Array &x) { return tanh(x); }, py::arg("x"));
  m.def("floor", [](const Array &x) { return floor(x); }, py::arg("x"));
  m.def("ceil", [](const Array &x) { return ceil(x); }, py::arg("x"));
  m.def("round", [](const Array &x) { return rint(x); }, py::arg("x"));
  m.def("sign", [](const Array &x) { return sign(x); }, py::arg("x"));
  m.def("isnan", [](const Array &x) { return isnan(x); }, py::arg("x"));
  m.def("isinf", [](const Array &x) { return isinf(x); }, py::arg("x"));
  m.def("isfinite", [](const Array &x) { return isfinite(x); }, py::arg("x"));
  m.def("exp2", [](const Array &x) { return exp2(x); }, py::arg("x"));
  m.def("expm1", [](const Array &x) { return expm1(x); }, py::arg("x"));
  m.def("log1p", [](const Array &x) { return log1p(x); }, py::arg("x"));
  m.def("cbrt", [](const Array &x) { return cbrt(x); }, py::arg("x"));
  m.def(
      "reciprocal", [](const Array &x) { return reciprocal(x); }, py::arg("x"));
  m.def("asinh", [](const Array &x) { return asinh(x); }, py::arg("x"));
  m.def("acosh", [](const Array &x) { return acosh(x); }, py::arg("x"));
  m.def("atanh", [](const Array &x) { return atanh(x); }, py::arg("x"));
  m.def("trunc", [](const Array &x) { return trunc(x); }, py::arg("x"));
  m.def("deg2rad", [](const Array &x) { return deg2rad(x); }, py::arg("x"));
  m.def("rad2deg", [](const Array &x) { return rad2deg(x); }, py::arg("x"));
  m.def(
      "where",
      [](const Array &cond, const Array &x, const Array &y) {
        return where(cond, x, y);
      },
      py::arg("cond"), py::arg("x"), py::arg("y"));

  // ====================================================================
  // Reduction
  // ====================================================================
  m.def(
      "sum",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return sum(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "mean",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return mean(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "max",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return max(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "min",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return min(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "prod",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return prod(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "argmax",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return argmax(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "argmin",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return argmin(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "any",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return any(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "all",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return all(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "var",
      [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return var(x, axis, keepdims, ddof);
      },
      py::arg("x"), py::arg("axis") = std::nullopt, py::arg("keepdims") = false,
      py::arg("ddof") = 0);
  m.def(
      "std",
      [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return ins::std(x, axis, keepdims, ddof);
      },
      py::arg("x"), py::arg("axis") = std::nullopt, py::arg("keepdims") = false,
      py::arg("ddof") = 0);
  m.def("cumsum", &cumsum, py::arg("x"), py::arg("axis"));
  m.def("cumprod", &cumprod, py::arg("x"), py::arg("axis"));
  m.def("cummax", &cummax, py::arg("x"), py::arg("axis"));
  m.def("cummin", &cummin, py::arg("x"), py::arg("axis"));
  m.def(
      "sem",
      [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return sem(x, axis, keepdims, ddof);
      },
      py::arg("x"), py::arg("axis") = std::nullopt, py::arg("keepdims") = false,
      py::arg("ddof") = 0);
  m.def(
      "count_nonzero",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return count_nonzero(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "median",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return median(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "quantile",
      [](const Array &x, double q, std::optional<int> axis, bool keepdims) {
        return quantile(x, q, axis, keepdims);
      },
      py::arg("x"), py::arg("q"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "quantile",
      [](const Array &x, const Array &q, std::optional<int> axis,
         bool keepdims) { return quantile(x, q, axis, keepdims); },
      py::arg("x"), py::arg("q"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "percentile",
      [](const Array &x, double q, std::optional<int> axis, bool keepdims) {
        return percentile(x, q, axis, keepdims);
      },
      py::arg("x"), py::arg("q"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "nansum",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return nansum(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "nanmean",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return nanmean(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "nanmax",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return nanmax(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "nanmin",
      [](const Array &x, std::optional<int> axis, bool keepdims) {
        return nanmin(x, axis, keepdims);
      },
      py::arg("x"), py::arg("axis") = std::nullopt,
      py::arg("keepdims") = false);
  m.def(
      "nanstd",
      [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return nanstd(x, axis, keepdims, ddof);
      },
      py::arg("x"), py::arg("axis") = std::nullopt, py::arg("keepdims") = false,
      py::arg("ddof") = 0);
  m.def(
      "nanvar",
      [](const Array &x, std::optional<int> axis, bool keepdims, int ddof) {
        return nanvar(x, axis, keepdims, ddof);
      },
      py::arg("x"), py::arg("axis") = std::nullopt, py::arg("keepdims") = false,
      py::arg("ddof") = 0);

  // ====================================================================
  // Manipulation
  // ====================================================================
  m.def("concat", &concat, py::arg("arrays"), py::arg("axis") = 0);
  m.def("stack", &stack, py::arg("arrays"), py::arg("axis") = 0);
  m.def(
      "vstack", [](const std::vector<Array> &t) { return vstack(t); },
      py::arg("arrays"));
  m.def(
      "hstack", [](const std::vector<Array> &t) { return hstack(t); },
      py::arg("arrays"));
  m.def(
      "split",
      [](const Array &x, int sections, int axis) {
        return split(x, sections, axis);
      },
      py::arg("x"), py::arg("sections"), py::arg("axis") = 0);
  m.def("tile", &tile, py::arg("x"), py::arg("reps"));
  m.def("repeat",
        py::overload_cast<const Array &, int, std::optional<int>>(&repeat),
        py::arg("x"), py::arg("repeats"), py::arg("axis") = std::nullopt);
  m.def("flip", &flip, py::arg("x"), py::arg("axis") = std::nullopt);
  m.def("pad", &pad, py::arg("x"), py::arg("pad_width"),
        py::arg("constant_value") = 0.0);
  m.def("reshape", &reshape, py::arg("x"), py::arg("new_shape"));
  m.def("flatten", &flatten, py::arg("x"));
  m.def("ravel", &ravel, py::arg("x"));
  m.def("squeeze", py::overload_cast<const Array &>(&squeeze), py::arg("x"));
  m.def("squeeze", py::overload_cast<const Array &, int>(&squeeze),
        py::arg("x"), py::arg("axis"));
  m.def("unsqueeze", &unsqueeze, py::arg("x"), py::arg("axis"));
  m.def("roll", &roll, py::arg("x"), py::arg("shift"),
        py::arg("axis") = std::nullopt);
  m.def("permute", &permute, py::arg("x"), py::arg("axes"));
  m.def("swapaxes", &swapaxes, py::arg("x"), py::arg("axis1"),
        py::arg("axis2"));
  m.def("moveaxis", &moveaxis, py::arg("x"), py::arg("source"),
        py::arg("destination"));
  m.def("fliplr", &fliplr, py::arg("x"));
  m.def("flipud", &flipud, py::arg("x"));
  m.def("rot90", &rot90, py::arg("x"), py::arg("k") = 1,
        py::arg("axes") = std::vector<int>{0, 1});
  m.def("diag", &diag, py::arg("x"), py::arg("k") = 0);
  m.def("diagonal", &diagonal, py::arg("x"), py::arg("offset") = 0,
        py::arg("axis1") = 0, py::arg("axis2") = 1);
  m.def("tril", &tril, py::arg("x"), py::arg("k") = 0);
  m.def("triu", &triu, py::arg("x"), py::arg("k") = 0);
  m.def("diff", &diff, py::arg("x"), py::arg("n") = 1, py::arg("axis") = -1);

  // ====================================================================
  // Linear Algebra
  // ====================================================================
  m.def("matmul", &matmul, py::arg("a"), py::arg("b"));
  m.def("dot", &dot, py::arg("a"), py::arg("b"));
  m.def("inner", &inner, py::arg("a"), py::arg("b"));
  m.def("outer", &outer, py::arg("a"), py::arg("b"));
  m.def("det", &det, py::arg("x"));
  m.def("slogdet", &slogdet, py::arg("x"));
  m.def("inv", &inv, py::arg("x"));
  m.def("pinv", &pinv, py::arg("x"), py::arg("rcond") = -1.0);
  m.def("solve", &solve, py::arg("a"), py::arg("b"));
  m.def("lstsq", &lstsq, py::arg("a"), py::arg("b"), py::arg("rcond") = -1.0);
  m.def("svd", &svd, py::arg("x"), py::arg("full_matrices") = true);
  m.def("svdvals", &svdvals, py::arg("x"));
  m.def("eigh", &eigh, py::arg("x"), py::arg("uplo") = "L");
  m.def("eigvalsh", &eigvalsh, py::arg("x"), py::arg("uplo") = "L");
  m.def("eig", &eig, py::arg("x"));
  m.def("eigvals", &eigvals, py::arg("x"));
  m.def("cholesky", &cholesky, py::arg("x"), py::arg("lower") = true);
  m.def("qr", &qr, py::arg("x"), py::arg("mode") = "reduced");
  m.def("lq", &lq, py::arg("x"), py::arg("mode") = "reduced");
  m.def("lu", &lu, py::arg("x"), py::arg("pivot") = true);
  m.def("norm", &norm, py::arg("x"), py::arg("ord") = 2.0);
  m.def("cond", &cond, py::arg("x"), py::arg("p") = 2.0);
  m.def("matrix_rank", &matrix_rank, py::arg("x"), py::arg("tol") = -1.0);
  m.def("matrix_power", &matrix_power, py::arg("x"), py::arg("n"));
  m.def("trace", &trace, py::arg("x"));
  m.def("cov", &cov, py::arg("x"), py::arg("rowvar") = true,
        py::arg("ddof") = 1);

  // ====================================================================
  // FFT (functions are in ins::fft:: namespace)
  // ====================================================================
  m.def(
      "fft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::fft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "ifft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::ifft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "fft2",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::fft2(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{-2, -1}, py::arg("norm") = "backward");
  m.def(
      "ifft2",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::ifft2(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{-2, -1}, py::arg("norm") = "backward");
  m.def(
      "fftn",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::fftn(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{}, py::arg("norm") = "backward");
  m.def(
      "ifftn",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::ifftn(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{}, py::arg("norm") = "backward");
  m.def(
      "rfft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::rfft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "irfft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::irfft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "fftfreq", [](int64_t n, double d) { return fft::fftfreq(n, d); },
      py::arg("n"), py::arg("d") = 1.0);
  m.def(
      "rfftfreq", [](int64_t n, double d) { return fft::rfftfreq(n, d); },
      py::arg("n"), py::arg("d") = 1.0);

  // ====================================================================
  // Complex
  // ====================================================================
  m.def("is_complex", static_cast<bool (*)(const Array &)>(&is_complex),
        py::arg("x"), "Check if array has complex dtype");
  m.def("has_complex_shape", &has_complex_shape, py::arg("x"),
        "Check if array uses legacy complex storage (last dim = 2)");
  m.def(
      "to_complex", [](const Array &real) { return to_complex(real); },
      py::arg("real"), "Convert real array to complex (imag = 0)");
  m.def(
      "to_complex",
      [](const Array &real, const Array &imag) {
        return to_complex(real, imag);
      },
      py::arg("real"), py::arg("imag"),
      "Convert real and imag arrays to complex");
  m.def("as_complex", &as_complex, py::arg("x"),
        "View real array with last dim=2 as complex (zero-copy)");
  m.def("as_real", &as_real, py::arg("x"),
        "View complex array as real with last dim=2 (zero-copy)");
  m.def(
      "real", [](const Array &z) { return real(z); }, py::arg("z"),
      "Extract real part from complex array");
  m.def(
      "imag", [](const Array &z) { return imag(z); }, py::arg("z"),
      "Extract imaginary part from complex array");

  // ====================================================================
  // Signal (ins::signal:: namespace)
  // ====================================================================
  {
    auto sig = m.def_submodule("signal", "Signal processing functions");

    // --- Result structs ---
    py::class_<signal::SpectralResult>(sig, "SpectralResult")
        .def_readonly("f", &signal::SpectralResult::f)
        .def_readonly("Pxx", &signal::SpectralResult::Pxx)
        .def("__repr__", [](const signal::SpectralResult &r) {
          return "<SpectralResult f=" + array_repr(r.f) +
                 " Pxx=" + array_repr(r.Pxx) + ">";
        });

    py::class_<signal::SpectrogramResult>(sig, "SpectrogramResult")
        .def_readonly("f", &signal::SpectrogramResult::f)
        .def_readonly("t", &signal::SpectrogramResult::t)
        .def_readonly("Sxx", &signal::SpectrogramResult::Sxx)
        .def("__repr__", [](const signal::SpectrogramResult &r) {
          return "<SpectrogramResult f=" + array_repr(r.f) +
                 " t=" + array_repr(r.t) + " Sxx=" + array_repr(r.Sxx) + ">";
        });

    py::class_<signal::GaussPulseResult>(sig, "GaussPulseResult")
        .def_readonly("yi", &signal::GaussPulseResult::yi)
        .def_readonly("yq", &signal::GaussPulseResult::yq)
        .def_readonly("ye", &signal::GaussPulseResult::ye)
        .def("__repr__", [](const signal::GaussPulseResult &r) {
          return "<GaussPulseResult yi=" + array_repr(r.yi) + ">";
        });

    // --- ChirpMethod enum ---
    py::enum_<signal::ChirpMethod>(sig, "ChirpMethod")
        .value("linear", signal::ChirpMethod::Linear)
        .value("quadratic", signal::ChirpMethod::Quadratic)
        .value("logarithmic", signal::ChirpMethod::Logarithmic)
        .value("hyperbolic", signal::ChirpMethod::Hyperbolic);

    // ----------------------------------------------------------------
    // Windows
    // ----------------------------------------------------------------
    sig.def("general_cosine", &signal::general_cosine, py::arg("M"),
            py::arg("a"), py::arg("sym") = true);
    sig.def(
        "get_window",
        [](const std::string &window, int64_t Nx, bool fftbins) {
          return signal::get_window(window, Nx, fftbins);
        },
        py::arg("window"), py::arg("Nx"), py::arg("fftbins") = true);
    sig.def(
        "get_window",
        [](const std::string &window, double param, int64_t Nx, bool fftbins) {
          return signal::get_window(window, param, Nx, fftbins);
        },
        py::arg("window"), py::arg("param"), py::arg("Nx"),
        py::arg("fftbins") = true);
    sig.def("boxcar", &signal::boxcar, py::arg("M"), py::arg("sym") = true);
    sig.def("triang", &signal::triang, py::arg("M"), py::arg("sym") = true);
    sig.def("parzen", &signal::parzen, py::arg("M"), py::arg("sym") = true);
    sig.def("bohman", &signal::bohman, py::arg("M"), py::arg("sym") = true);
    sig.def("bartlett", &signal::bartlett, py::arg("M"), py::arg("sym") = true);
    sig.def("cosine", &signal::cosine, py::arg("M"), py::arg("sym") = true);
    sig.def("exponential", &signal::exponential, py::arg("M"),
            py::arg("center") = -1.0, py::arg("tau") = 1.0,
            py::arg("sym") = true);
    sig.def("blackman", &signal::blackman, py::arg("M"), py::arg("sym") = true);
    sig.def("nuttall", &signal::nuttall, py::arg("M"), py::arg("sym") = true);
    sig.def("blackmanharris", &signal::blackmanharris, py::arg("M"),
            py::arg("sym") = true);
    sig.def("flattop", &signal::flattop, py::arg("M"), py::arg("sym") = true);
    sig.def("hann", &signal::hann, py::arg("M"), py::arg("sym") = true);
    sig.def("general_hamming", &signal::general_hamming, py::arg("M"),
            py::arg("alpha"), py::arg("sym") = true);
    sig.def("hamming", &signal::hamming, py::arg("M"), py::arg("sym") = true);
    sig.def("tukey", &signal::tukey, py::arg("M"), py::arg("alpha") = 0.5,
            py::arg("sym") = true);
    sig.def("barthann", &signal::barthann, py::arg("M"), py::arg("sym") = true);
    sig.def("kaiser", &signal::kaiser, py::arg("M"), py::arg("beta"),
            py::arg("sym") = true);
    sig.def("gaussian", &signal::gaussian, py::arg("M"), py::arg("std"),
            py::arg("sym") = true);
    sig.def("general_gaussian", &signal::general_gaussian, py::arg("M"),
            py::arg("p"), py::arg("sig"), py::arg("sym") = true);
    sig.def("chebwin", &signal::chebwin, py::arg("M"), py::arg("at"),
            py::arg("sym") = true);
    sig.def("taylor", &signal::taylor, py::arg("M"), py::arg("nbar") = 4,
            py::arg("sll") = -30.0, py::arg("norm") = true,
            py::arg("sym") = true);

    // ----------------------------------------------------------------
    // Waveforms
    // ----------------------------------------------------------------
    sig.def("sawtooth", &signal::sawtooth, py::arg("t"),
            py::arg("width") = 1.0);
    sig.def("square_wf", &signal::square, py::arg("t"), py::arg("duty") = 0.5);
    sig.def("square", &signal::square, py::arg("t"), py::arg("duty") = 0.5);
    sig.def("gausspulse", &signal::gausspulse, py::arg("t"),
            py::arg("fc") = 1000, py::arg("bw") = 0.5, py::arg("bwr") = -6,
            py::arg("tpr") = -60);
    sig.def("gausspulse_full", &signal::gausspulse_full, py::arg("t"),
            py::arg("fc") = 1000, py::arg("bw") = 0.5, py::arg("bwr") = -6,
            py::arg("tpr") = -60);
    sig.def("chirp", &signal::chirp, py::arg("t"), py::arg("f0"), py::arg("t1"),
            py::arg("f1"), py::arg("method") = signal::ChirpMethod::Linear,
            py::arg("phi") = 0.0, py::arg("vertex_zero") = true);
    sig.def("unit_impulse", &signal::unit_impulse, py::arg("shape"),
            py::arg("idx") = -1, py::arg("dtype") = DType::F64,
            py::arg("place") = get_device());

    // ----------------------------------------------------------------
    // B-Splines
    // ----------------------------------------------------------------
    sig.def("gauss_spline", &signal::gauss_spline, py::arg("x"), py::arg("n"));
    sig.def("cubic", &signal::cubic, py::arg("x"));
    sig.def("quadratic", &signal::quadratic, py::arg("x"));

    // ----------------------------------------------------------------
    // Filter Design
    // ----------------------------------------------------------------
    sig.def("kaiser_beta", &signal::kaiser_beta, py::arg("a"));
    sig.def("kaiser_atten", &signal::kaiser_atten, py::arg("numtaps"),
            py::arg("width"));
    sig.def("firwin", &signal::firwin, py::arg("numtaps"), py::arg("cutoff"),
            py::arg("window") = "hamming", py::arg("pass_zero") = "lowpass",
            py::arg("scale") = true);
    sig.def("firwin2", &signal::firwin2, py::arg("numtaps"), py::arg("freq"),
            py::arg("gain"), py::arg("nfreqs") = 0,
            py::arg("window") = "hamming", py::arg("antisymmetric") = false);
    sig.def("cmplx_sort", &signal::cmplx_sort, py::arg("p"));

    // ----------------------------------------------------------------
    // Convolution
    // ----------------------------------------------------------------
    sig.def("fftconvolve", &signal::fftconvolve, py::arg("in1"), py::arg("in2"),
            py::arg("mode") = "full");
    sig.def("correlate", &signal::correlate, py::arg("in1"), py::arg("in2"),
            py::arg("mode") = "full");
    sig.def("convolve2d", &signal::convolve2d, py::arg("in1"), py::arg("in2"),
            py::arg("mode") = "full");
    sig.def("correlate2d", &signal::correlate2d, py::arg("in1"), py::arg("in2"),
            py::arg("mode") = "full");
    sig.def("choose_conv_method", &signal::choose_conv_method, py::arg("in1"),
            py::arg("in2"), py::arg("mode") = "full");
    sig.def("correlation_lags", &signal::correlation_lags, py::arg("in1_len"),
            py::arg("in2_len"), py::arg("mode") = "full");

    // ----------------------------------------------------------------
    // Filtering
    // ----------------------------------------------------------------
    sig.def("hilbert", &signal::hilbert, py::arg("x"), py::arg("N") = -1);
    sig.def("hilbert2", &signal::hilbert2, py::arg("x"), py::arg("N") = -1);
    sig.def("detrend", &signal::detrend, py::arg("data"), py::arg("axis") = -1,
            py::arg("type") = "linear");
    sig.def("wiener", &signal::wiener, py::arg("im"),
            py::arg("mysize") = std::vector<int64_t>{},
            py::arg("noise") = -1.0);
    sig.def("firfilter", &signal::firfilter, py::arg("b"), py::arg("x"),
            py::arg("axis") = -1);
    sig.def("lfilter", &signal::lfilter, py::arg("b"), py::arg("a"),
            py::arg("x"), py::arg("axis") = -1);
    sig.def("lfilter_zi", &signal::lfilter_zi, py::arg("b"), py::arg("a"));
    sig.def("filtfilt", &signal::filtfilt, py::arg("b"), py::arg("a"),
            py::arg("x"), py::arg("axis") = -1);
    sig.def("decimate", &signal::decimate, py::arg("x"), py::arg("q"),
            py::arg("axis") = -1, py::arg("zero_phase") = true);
    sig.def("resample", &signal::resample, py::arg("x"), py::arg("num"),
            py::arg("axis") = -1);
    sig.def("resample_poly", &signal::resample_poly, py::arg("x"),
            py::arg("up"), py::arg("down"), py::arg("axis") = -1);
    sig.def("freq_shift", &signal::freq_shift, py::arg("x"), py::arg("freq"),
            py::arg("fs"));

    // ----------------------------------------------------------------
    // Spectral Analysis
    // ----------------------------------------------------------------
    sig.def("csd", &signal::csd, py::arg("x"), py::arg("y"),
            py::arg("fs") = 1.0, py::arg("window") = "hann",
            py::arg("nperseg") = 256, py::arg("noverlap") = 0,
            py::arg("nfft") = 0, py::arg("detrend") = "constant",
            py::arg("return_onesided") = true, py::arg("scaling") = "density");
    sig.def("welch", &signal::welch, py::arg("x"), py::arg("fs") = 1.0,
            py::arg("window") = "hann", py::arg("nperseg") = 256,
            py::arg("noverlap") = 0, py::arg("nfft") = 0,
            py::arg("detrend") = "constant", py::arg("return_onesided") = true,
            py::arg("scaling") = "density");
    sig.def("periodogram", &signal::periodogram, py::arg("x"),
            py::arg("fs") = 1.0, py::arg("window") = "boxcar",
            py::arg("nfft") = 0, py::arg("detrend") = "constant",
            py::arg("return_onesided") = true, py::arg("scaling") = "density");
    sig.def("coherence", &signal::coherence, py::arg("x"), py::arg("y"),
            py::arg("fs") = 1.0, py::arg("window") = "hann",
            py::arg("nperseg") = 256, py::arg("noverlap") = 0,
            py::arg("nfft") = 0, py::arg("detrend") = "constant");
    sig.def("spectrogram", &signal::spectrogram, py::arg("x"),
            py::arg("fs") = 1.0, py::arg("window") = "hann",
            py::arg("nperseg") = 256, py::arg("noverlap") = 0,
            py::arg("nfft") = 0, py::arg("detrend") = "constant",
            py::arg("return_onesided") = true, py::arg("mode") = "psd");
    sig.def("stft", &signal::stft, py::arg("x"), py::arg("fs") = 1.0,
            py::arg("window") = "hann", py::arg("nperseg") = 256,
            py::arg("noverlap") = 0, py::arg("nfft") = 0);
    sig.def("vectorstrength", &signal::vectorstrength, py::arg("events"),
            py::arg("period"));

    // ----------------------------------------------------------------
    // Wavelets
    // ----------------------------------------------------------------
    sig.def("morlet", &signal::morlet, py::arg("M"), py::arg("w") = 5.0,
            py::arg("s") = 1.0, py::arg("complete") = true);
    sig.def("ricker", &signal::ricker, py::arg("points"), py::arg("a"));
    sig.def("morlet2", &signal::morlet2, py::arg("M"), py::arg("s"),
            py::arg("w") = 5.0);
    sig.def(
        "cwt",
        [](const Array &data, std::function<Array(int64_t, double)> wavelet,
           const std::vector<double> &widths) {
          return signal::cwt(data, wavelet, widths);
        },
        py::arg("data"), py::arg("wavelet"), py::arg("widths"));

    // ----------------------------------------------------------------
    // Acoustics
    // ----------------------------------------------------------------
    sig.def("mel2hz", &signal::mel2hz, py::arg("mels"));
    sig.def("hz2mel", &signal::hz2mel, py::arg("hz"));
    sig.def("mel_frequencies", &signal::mel_frequencies, py::arg("n_mels"),
            py::arg("f_low") = 0.0, py::arg("f_high") = 11000.0);
    sig.def("hz2bark", &signal::hz2bark, py::arg("hz"));
    sig.def("bark2hz", &signal::bark2hz, py::arg("bark"));

    // ----------------------------------------------------------------
    // Demod
    // ----------------------------------------------------------------
    sig.def("fm_demod", &signal::fm_demod, py::arg("x"), py::arg("axis") = -1);

    // ----------------------------------------------------------------
    // Peak Finding
    // ----------------------------------------------------------------
    sig.def(
        "argrelextrema",
        [](const Array &data, const std::string &comparator, int axis,
           int order, const std::string &mode) {
          return signal::argrelextrema(data, comparator, axis, order, mode);
        },
        py::arg("data"), py::arg("comparator"), py::arg("axis") = 0,
        py::arg("order") = 1, py::arg("mode") = "clip");
    sig.def(
        "argrelmax",
        [](const Array &data, int axis, int order, const std::string &mode) {
          return signal::argrelmax(data, axis, order, mode);
        },
        py::arg("data"), py::arg("axis") = 0, py::arg("order") = 1,
        py::arg("mode") = "clip");
    sig.def(
        "argrelmin",
        [](const Array &data, int axis, int order, const std::string &mode) {
          return signal::argrelmin(data, axis, order, mode);
        },
        py::arg("data"), py::arg("axis") = 0, py::arg("order") = 1,
        py::arg("mode") = "clip");

    // ----------------------------------------------------------------
    // Radar
    // ----------------------------------------------------------------
    sig.def("pulse_compression", &signal::pulse_compression, py::arg("x"),
            py::arg("template_tx"), py::arg("normalize") = false,
            py::arg("window") = "", py::arg("nfft") = 0);
    sig.def("pulse_doppler", &signal::pulse_doppler, py::arg("x"),
            py::arg("window") = "", py::arg("nfft") = 0);
    sig.def("cfar_alpha", &signal::cfar_alpha, py::arg("pfa"), py::arg("N"));
    sig.def(
        "ca_cfar",
        [](const Array &data, const std::vector<int> &guard_cells,
           const std::vector<int> &reference_cells, double pfa) {
          return signal::ca_cfar(data, guard_cells, reference_cells, pfa);
        },
        py::arg("data"), py::arg("guard_cells"), py::arg("reference_cells"),
        py::arg("pfa") = 1e-3);
    sig.def("mvdr", &signal::mvdr, py::arg("x"), py::arg("sv"),
            py::arg("calc_cov") = true);
    sig.def("ambgfun", &signal::ambgfun, py::arg("x"), py::arg("fs"),
            py::arg("prf"), py::arg("y") = Array(), py::arg("cut") = "2d",
            py::arg("cutValue") = 0);

    // ----------------------------------------------------------------
    // Estimation (KalmanFilter)
    // ----------------------------------------------------------------
    py::class_<signal::KalmanFilter>(sig, "KalmanFilter")
        .def(py::init<int, int, int, int, DType>(), py::arg("dim_x"),
             py::arg("dim_z"), py::arg("dim_u") = 0, py::arg("points") = 1,
             py::arg("dtype") = DType::F64)
        .def_readwrite("x", &signal::KalmanFilter::x)
        .def_readwrite("P", &signal::KalmanFilter::P)
        .def_readwrite("z", &signal::KalmanFilter::z)
        .def_readwrite("R", &signal::KalmanFilter::R)
        .def_readwrite("Q", &signal::KalmanFilter::Q)
        .def_readwrite("F", &signal::KalmanFilter::F)
        .def_readwrite("H", &signal::KalmanFilter::H)
        .def_readonly("dim_x", &signal::KalmanFilter::dim_x)
        .def_readonly("dim_z", &signal::KalmanFilter::dim_z)
        .def_readonly("points", &signal::KalmanFilter::points)
        .def("predict", &signal::KalmanFilter::predict)
        .def("update", &signal::KalmanFilter::update, py::arg("z"));

    // ----------------------------------------------------------------
    // Signal I/O
    // ----------------------------------------------------------------
    sig.def("read_bin", &signal::read_bin, py::arg("file"),
            py::arg("dtype") = DType::U8, py::arg("num_samples") = 0,
            py::arg("offset") = 0);
    sig.def("write_bin", &signal::write_bin, py::arg("file"), py::arg("data"),
            py::arg("append") = true);
    sig.def("unpack_bin", &signal::unpack_bin, py::arg("binary"),
            py::arg("dtype"), py::arg("endianness") = "L");
    sig.def("pack_bin", &signal::pack_bin, py::arg("data"));
    sig.def("read_sigmf", &signal::read_sigmf, py::arg("data_file"),
            py::arg("meta_file") = "", py::arg("num_samples") = 0,
            py::arg("offset") = 0);
    sig.def("write_sigmf", &signal::write_sigmf, py::arg("data_file"),
            py::arg("data"), py::arg("append") = true);

    // ----------------------------------------------------------------
    // Top-level signal functions (in ins:: namespace, aliased here)
    // ----------------------------------------------------------------
    sig.def("convolve", &convolve, py::arg("a"), py::arg("v"),
            py::arg("mode") = "full");
    sig.def("unwrap", &unwrap, py::arg("p"), py::arg("axis") = -1,
            py::arg("discont") = M_PI, py::arg("period") = 2 * M_PI);
    sig.def("sinc", &sinc, py::arg("x"));
  }

  // Top-level aliases (scipy-compatible)
  m.def("convolve", &convolve, py::arg("a"), py::arg("v"),
        py::arg("mode") = "full");
  m.def("unwrap", &unwrap, py::arg("p"), py::arg("axis") = -1,
        py::arg("discont") = M_PI, py::arg("period") = 2 * M_PI);
  m.def("sinc", &sinc, py::arg("x"));

  // ====================================================================
  // Plot (ins::plot:: namespace, requires INSIGHT_USE_MATPLOT)
  // ====================================================================
#ifdef INSIGHT_USE_MATPLOT
  {
    auto plt = m.def_submodule("plot", "Plotting functions (matplotlib-style)");

    // Figure & Axes
    plt.def("figure", &plot::figure, py::arg("number") = -1);
    plt.def("subplot", &plot::subplot, py::arg("rows"), py::arg("cols"),
            py::arg("index"));
    plt.def("hold", &plot::hold, py::arg("on"));
    plt.def("clf", &plot::clf);
    plt.def("show", &plot::show);
    plt.def("save", &plot::save, py::arg("filename"));

    // Line plots
    plt.def(
        "plot",
        [](const Array &y, const std::string &fmt) { plot::plot(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "plot",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::plot(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");
    plt.def(
        "plot3",
        [](const Array &x, const Array &y, const Array &z,
           const std::string &fmt) { plot::plot3(x, y, z, fmt); },
        py::arg("x"), py::arg("y"), py::arg("z"), py::arg("format") = "");
    plt.def(
        "fplot",
        [](std::function<double(double)> f, double xmin, double xmax,
           const std::string &fmt,
           int n) { plot::fplot(f, xmin, xmax, fmt, n); },
        py::arg("f"), py::arg("xmin"), py::arg("xmax"), py::arg("format") = "",
        py::arg("n_points") = 100);
    plt.def(
        "stairs",
        [](const Array &y, const std::string &fmt) { plot::stairs(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "stairs",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::stairs(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");
    plt.def("errorbar", &plot::errorbar, py::arg("x"), py::arg("y"),
            py::arg("err"), py::arg("format") = "");
    plt.def("area", [](const Array &y) { plot::area(y); }, py::arg("y"));
    plt.def(
        "area", [](const Array &x, const Array &y) { plot::area(x, y); },
        py::arg("x"), py::arg("y"));
    plt.def("fill", &plot::fill, py::arg("x"), py::arg("y"),
            py::arg("color") = "b");
    plt.def("line", &plot::line, py::arg("x"), py::arg("y"),
            py::arg("format") = "");

    // Logarithmic
    plt.def(
        "semilogx",
        [](const Array &y, const std::string &fmt) { plot::semilogx(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "semilogx",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::semilogx(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");
    plt.def(
        "semilogy",
        [](const Array &y, const std::string &fmt) { plot::semilogy(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "semilogy",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::semilogy(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");
    plt.def(
        "loglog",
        [](const Array &y, const std::string &fmt) { plot::loglog(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "loglog",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::loglog(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");

    // Scatter
    plt.def("scatter", &plot::scatter, py::arg("x"), py::arg("y"),
            py::arg("size") = 20.0);
    plt.def("scatter3", &plot::scatter3, py::arg("x"), py::arg("y"),
            py::arg("z"), py::arg("size") = 20.0);
    plt.def("binscatter", &plot::binscatter, py::arg("x"), py::arg("y"));

    // Bar
    plt.def(
        "bar", [](const Array &y, double w) { plot::bar(y, w); }, py::arg("y"),
        py::arg("width") = 0.8);
    plt.def(
        "bar",
        [](const Array &x, const Array &y, double w) { plot::bar(x, y, w); },
        py::arg("x"), py::arg("y"), py::arg("width") = 0.8);
    plt.def(
        "barh", [](const Array &y, double w) { plot::barh(y, w); },
        py::arg("y"), py::arg("width") = 0.8);
    plt.def(
        "barh",
        [](const Array &x, const Array &y, double w) { plot::barh(x, y, w); },
        py::arg("x"), py::arg("y"), py::arg("width") = 0.8);
    plt.def("barstacked", py::overload_cast<const Array &>(&plot::barstacked),
            py::arg("Y"));
    plt.def("barstacked",
            py::overload_cast<const Array &, const Array &>(&plot::barstacked),
            py::arg("x"), py::arg("Y"));
    plt.def("pareto", [](const Array &y) { plot::pareto(y); }, py::arg("y"));
    plt.def(
        "pareto", [](const Array &x, const Array &y) { plot::pareto(x, y); },
        py::arg("x"), py::arg("y"));

    // Histograms
    plt.def("hist", &plot::hist, py::arg("data"), py::arg("bins") = 10);
    plt.def("hist2", &plot::hist2, py::arg("x"), py::arg("y"),
            py::arg("bins") = 10);
    plt.def(
        "boxplot", [](const Array &d) { plot::boxplot(d); }, py::arg("data"));
    plt.def(
        "boxplot",
        [](const Array &d, const std::vector<std::string> &g) {
          plot::boxplot(d, g);
        },
        py::arg("data"), py::arg("groups"));
    plt.def("heatmap", &plot::heatmap, py::arg("data"));

    // Stem
    plt.def(
        "stem",
        [](const Array &y, const std::string &fmt) { plot::stem(y, fmt); },
        py::arg("y"), py::arg("format") = "");
    plt.def(
        "stem",
        [](const Array &x, const Array &y, const std::string &fmt) {
          plot::stem(x, y, fmt);
        },
        py::arg("x"), py::arg("y"), py::arg("format") = "");
    plt.def("stem3", &plot::stem3, py::arg("x"), py::arg("y"), py::arg("z"),
            py::arg("format") = "");

    // Contour
    plt.def("contour", &plot::contour, py::arg("X"), py::arg("Y"), py::arg("Z"),
            py::arg("levels") = 10);
    plt.def("contourf", &plot::contourf, py::arg("X"), py::arg("Y"),
            py::arg("Z"), py::arg("levels") = 10);
    plt.def("pcolor", &plot::pcolor, py::arg("C"));

    // Surface / Mesh
    plt.def("surf", &plot::surf, py::arg("X"), py::arg("Y"), py::arg("Z"));
    plt.def("surfc", &plot::surfc, py::arg("X"), py::arg("Y"), py::arg("Z"));
    plt.def("mesh", &plot::mesh, py::arg("X"), py::arg("Y"), py::arg("Z"));
    plt.def("meshc", &plot::meshc, py::arg("X"), py::arg("Y"), py::arg("Z"));
    plt.def("meshz", &plot::meshz, py::arg("X"), py::arg("Y"), py::arg("Z"));
    plt.def("waterfall", &plot::waterfall, py::arg("X"), py::arg("Y"),
            py::arg("Z"));
    plt.def(
        "fsurf",
        [](std::function<double(double, double)> f,
           const std::vector<double> &uv) { plot::fsurf(f, uv); },
        py::arg("f"), py::arg("uv") = std::vector<double>{-5, 5, -5, 5});
    plt.def(
        "fmesh",
        [](std::function<double(double, double)> f,
           const std::vector<double> &uv) { plot::fmesh(f, uv); },
        py::arg("f"), py::arg("uv") = std::vector<double>{-5, 5, -5, 5});
    plt.def("ribbon", [](const Array &y) { plot::ribbon(y); }, py::arg("y"));
    plt.def(
        "ribbon", [](const Array &x, const Array &y) { plot::ribbon(x, y); },
        py::arg("x"), py::arg("y"));
    plt.def("fence", &plot::fence, py::arg("X"), py::arg("Y"), py::arg("Z"));

    // Vector fields
    plt.def("quiver", &plot::quiver, py::arg("X"), py::arg("Y"), py::arg("U"),
            py::arg("V"), py::arg("scale") = 1.0);
    plt.def("quiver3", &plot::quiver3, py::arg("X"), py::arg("Y"), py::arg("Z"),
            py::arg("U"), py::arg("V"), py::arg("W"));
    plt.def("feather", &plot::feather, py::arg("u"), py::arg("v"));

    // Polar
    plt.def("polarplot", &plot::polarplot, py::arg("theta"), py::arg("rho"),
            py::arg("format") = "");
    plt.def("polarscatter", &plot::polarscatter, py::arg("theta"),
            py::arg("rho"), py::arg("size") = 20.0);
    plt.def("polarhistogram", &plot::polarhistogram, py::arg("data"),
            py::arg("bins") = 10);
    plt.def("compass", &plot::compass, py::arg("u"), py::arg("v"));
    plt.def("ezpolar", &plot::ezpolar, py::arg("expr"));

    // Pie
    plt.def("pie", [](const Array &x) { plot::pie(x); }, py::arg("x"));
    plt.def(
        "pie",
        [](const Array &x, const std::vector<std::string> &labels) {
          plot::pie(x, labels);
        },
        py::arg("x"), py::arg("labels"));

    // Images
    plt.def("imshow", &plot::imshow, py::arg("data"));
    plt.def("image", &plot::image, py::arg("data"));
    plt.def("imagesc", &plot::imagesc, py::arg("data"));

    // Graph
    plt.def("graph", &plot::graph, py::arg("sources"), py::arg("targets"));
    plt.def("digraph", &plot::digraph, py::arg("sources"), py::arg("targets"));

    // Misc
    plt.def("parallelplot", &plot::parallelplot, py::arg("data"));
    plt.def("plotmatrix", &plot::plotmatrix, py::arg("X"));
    plt.def("wordcloud", &plot::wordcloud, py::arg("words"), py::arg("sizes"));
    plt.def("rgbplot", &plot::rgbplot, py::arg("data"));

    // Labels & Titles
    plt.def("title", &plot::title, py::arg("text"));
    plt.def("subtitle", &plot::subtitle, py::arg("text"));
    plt.def("xlabel", &plot::xlabel, py::arg("text"));
    plt.def("ylabel", &plot::ylabel, py::arg("text"));
    plt.def("zlabel", &plot::zlabel, py::arg("text"));
    plt.def("legend", &plot::legend, py::arg("labels"));

    // Axis limits
    plt.def("xlim", &plot::xlim, py::arg("xmin"), py::arg("xmax"));
    plt.def("ylim", &plot::ylim, py::arg("ymin"), py::arg("ymax"));
    plt.def("zlim", &plot::zlim, py::arg("zmin"), py::arg("zmax"));
    plt.def("axis_equal", &plot::axis_equal);
    plt.def("axis_tight", &plot::axis_tight);
    plt.def("axis_off", &plot::axis_off);
    plt.def("axis_ij", &plot::axis_ij);
    plt.def("grid", &plot::grid, py::arg("on"));

    // Ticks
    plt.def("xticks", &plot::xticks, py::arg("ticks"));
    plt.def("yticks", &plot::yticks, py::arg("ticks"));
    plt.def("zticks", &plot::zticks, py::arg("ticks"));
    plt.def("xticklabels", &plot::xticklabels, py::arg("labels"));
    plt.def("yticklabels", &plot::yticklabels, py::arg("labels"));
    plt.def("xtickangle", &plot::xtickangle, py::arg("angle"));
    plt.def("ytickangle", &plot::ytickangle, py::arg("angle"));

    // Colormap
    py::enum_<plot::Colormap>(plt, "Colormap")
        .value("parula", plot::Colormap::Parula)
        .value("jet", plot::Colormap::Jet)
        .value("hsv", plot::Colormap::HSV)
        .value("hot", plot::Colormap::Hot)
        .value("cool", plot::Colormap::Cool)
        .value("spring", plot::Colormap::Spring)
        .value("summer", plot::Colormap::Summer)
        .value("autumn", plot::Colormap::Autumn)
        .value("winter", plot::Colormap::Winter)
        .value("bone", plot::Colormap::Bone)
        .value("copper", plot::Colormap::Copper)
        .value("pink", plot::Colormap::Pink)
        .value("lines", plot::Colormap::Lines)
        .value("colorcube", plot::Colormap::Colorcube)
        .value("prism", plot::Colormap::Prism)
        .value("flag", plot::Colormap::Flag);
    plt.def("colormap", &plot::colormap, py::arg("cmap"));
    plt.def("colorbar", &plot::colorbar, py::arg("on") = true);

    // Annotations
    plt.def(
        "text",
        [](double x, double y, const std::string &s) { plot::text(x, y, s); },
        py::arg("x"), py::arg("y"), py::arg("text"));
    plt.def("arrow", &plot::arrow, py::arg("x1"), py::arg("y1"), py::arg("x2"),
            py::arg("y2"));
    plt.def("rectangle", &plot::rectangle, py::arg("x"), py::arg("y"),
            py::arg("w"), py::arg("h"));
    plt.def("ellipse", &plot::ellipse, py::arg("cx"), py::arg("cy"),
            py::arg("rx"), py::arg("ry"));

    // 3D Camera
    plt.def("view", &plot::view, py::arg("azimuth"), py::arg("elevation"));
  }
#endif // INSIGHT_USE_MATPLOT

  // ====================================================================
  // Random
  // ====================================================================
  m.def(
      "rand",
      [](const Shape &shape, DType dtype, const Place &place) {
        return rand(shape, dtype, place);
      },
      py::arg("shape"), py::arg("dtype") = DType::F32,
      py::arg("place") = get_device());
  m.def(
      "randn",
      [](const Shape &shape, DType dtype, const Place &place) {
        return randn(shape, dtype, place);
      },
      py::arg("shape"), py::arg("dtype") = DType::F32,
      py::arg("place") = get_device());
  m.def(
      "randint",
      [](int64_t low, int64_t high, const Shape &shape, DType dtype,
         const Place &place) {
        return randint(low, high, shape, dtype, place);
      },
      py::arg("low"), py::arg("high"), py::arg("shape"),
      py::arg("dtype") = DType::I64, py::arg("place") = get_device());
  m.def(
      "normal",
      [](double mean, double std_val, const Shape &shape, DType dtype,
         const Place &place) {
        return normal(mean, std_val, shape, dtype, place);
      },
      py::arg("mean") = 0.0, py::arg("std") = 1.0, py::arg("shape"),
      py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def(
      "uniform",
      [](double low, double high, const Shape &shape, DType dtype,
         const Place &place) {
        return uniform(low, high, shape, dtype, place);
      },
      py::arg("low") = 0.0, py::arg("high") = 1.0, py::arg("shape"),
      py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def(
      "randperm",
      [](int64_t n, DType dtype, const Place &place) {
        return randperm(n, dtype, place);
      },
      py::arg("n"), py::arg("dtype") = DType::I64,
      py::arg("place") = get_device());

  // ====================================================================
  // Cast
  // ====================================================================
  m.def(
      "cast",
      [](const Array &x, DType dtype, bool copy) {
        return cast(x, dtype, copy);
      },
      py::arg("x"), py::arg("dtype"), py::arg("copy") = true);

  // ====================================================================
  // Indexing
  // ====================================================================
  m.def("take", &take, py::arg("x"), py::arg("indices"),
        py::arg("axis") = std::nullopt);
  m.def("nonzero", &nonzero, py::arg("x"));
  m.def("flatnonzero", &flatnonzero, py::arg("x"));
  m.def("argsort", &argsort, py::arg("x"), py::arg("axis") = -1,
        py::arg("descending") = false);
  m.def("sort", &sort, py::arg("x"), py::arg("axis") = -1,
        py::arg("descending") = false);
  m.def("masked_select", &masked_select, py::arg("x"), py::arg("mask"));
  m.def(
      "searchsorted",
      [](const Array &x, const Array &v, const std::string &side) {
        return searchsorted(x, v, side);
      },
      py::arg("x"), py::arg("v"), py::arg("side") = "left");

  // --- Additional indexing ---
  {
    py::class_<UniqueResult>(m, "UniqueResult")
        .def_readonly("unique", &UniqueResult::unique)
        .def_readonly("indices", &UniqueResult::indices)
        .def_readonly("inverse", &UniqueResult::inverse)
        .def_readonly("counts", &UniqueResult::counts);
    m.def("unique", &unique, py::arg("x"), py::arg("return_indices") = false,
          py::arg("return_inverse") = false, py::arg("return_counts") = false);
    m.def("topk", &topk, py::arg("x"), py::arg("k"), py::arg("axis") = -1,
          py::arg("largest") = true, py::arg("sorted") = true);
    m.def("gather", &gather, py::arg("x"), py::arg("dim"), py::arg("index"));
    m.def("scatter", &scatter, py::arg("x"), py::arg("dim"), py::arg("index"),
          py::arg("src"));
    m.def("scatter_add", &scatter_add, py::arg("x"), py::arg("dim"),
          py::arg("index"), py::arg("src"));
    m.def("scatter_reduce", &scatter_reduce, py::arg("x"), py::arg("dim"),
          py::arg("index"), py::arg("src"), py::arg("reduce") = "replace");
    m.def("interp", &interp, py::arg("x"), py::arg("xp"), py::arg("fp"),
          py::arg("left") = std::nullopt, py::arg("right") = std::nullopt);
    m.def(
        "indices",
        [](const Shape &shape, bool sparse) { return indices(shape, sparse); },
        py::arg("shape"), py::arg("sparse") = false);
    m.def("ix_", &ix_, py::arg("arrays"));
  }

  // --- Additional random ---
  m.def("seed", &seed, py::arg("seed"));
  m.def("get_seed", &get_seed);
  m.def("rand_like", &rand_like, py::arg("x"));
  m.def("randn_like", &randn_like, py::arg("x"));
  m.def(
      "exponential",
      [](double scale, const Shape &shape, DType dtype, const Place &place) {
        return exponential(scale, shape, dtype, place);
      },
      py::arg("scale"), py::arg("shape"), py::arg("dtype") = DType::F32,
      py::arg("place") = get_device());
  m.def(
      "gamma",
      [](double shape_p, double rate, const Shape &shape, DType dtype,
         const Place &place) {
        return gamma(shape_p, rate, shape, dtype, place);
      },
      py::arg("shape"), py::arg("rate"), py::arg("out_shape"),
      py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def(
      "beta",
      [](double a, double b, const Shape &shape, DType dtype,
         const Place &place) { return beta(a, b, shape, dtype, place); },
      py::arg("a"), py::arg("b"), py::arg("shape"),
      py::arg("dtype") = DType::F32, py::arg("place") = get_device());
  m.def(
      "binomial",
      [](int64_t n, double p, const Shape &shape, DType dtype,
         const Place &place) { return binomial(n, p, shape, dtype, place); },
      py::arg("n"), py::arg("p"), py::arg("shape"),
      py::arg("dtype") = DType::I64, py::arg("place") = get_device());
  m.def(
      "poisson",
      [](double lam, const Shape &shape, DType dtype, const Place &place) {
        return poisson(lam, shape, dtype, place);
      },
      py::arg("lam"), py::arg("shape"), py::arg("dtype") = DType::I64,
      py::arg("place") = get_device());

  // --- Additional FFT ---
  m.def(
      "fftshift",
      [](const Array &x, int axis) { return fft::fftshift(x, axis); },
      py::arg("x"), py::arg("axis") = -1);
  m.def(
      "ifftshift",
      [](const Array &x, int axis) { return fft::ifftshift(x, axis); },
      py::arg("x"), py::arg("axis") = -1);
  m.def(
      "next_fast_len", [](int target) { return fft::next_fast_len(target); },
      py::arg("target"));
  m.def(
      "hfft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::hfft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "ihfft",
      [](const Array &x, int n, int axis, const std::string &norm) {
        return fft::ihfft(x, n, axis, norm);
      },
      py::arg("x"), py::arg("n") = -1, py::arg("axis") = -1,
      py::arg("norm") = "backward");
  m.def(
      "rfft2",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::rfft2(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{-2, -1}, py::arg("norm") = "backward");
  m.def(
      "irfft2",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::irfft2(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{-2, -1}, py::arg("norm") = "backward");
  m.def(
      "rfftn",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::rfftn(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{}, py::arg("norm") = "backward");
  m.def(
      "irfftn",
      [](const Array &x, const std::vector<int64_t> &s,
         const std::vector<int> &axes,
         const std::string &norm) { return fft::irfftn(x, s, axes, norm); },
      py::arg("x"), py::arg("s") = std::vector<int64_t>{},
      py::arg("axes") = std::vector<int>{}, py::arg("norm") = "backward");
}
