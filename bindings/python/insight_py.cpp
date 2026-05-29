// bindings/python/insight_py.cpp
//
// Python bindings for the Insight7 scientific computing framework.
// Built with pybind11, wrapping the C++ ins:: namespace API.
// API style: PaddlePaddle (ins.float32, ins.CPUPlace(), etc.)

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  std::ostringstream ss;
  ss << "<insight.Array shape=(";
  for (int i = 0; i < a.shape().ndim(); i++) {
    if (i > 0)
      ss << ", ";
    ss << a.shape().dim(i);
  }
  ss << ") dtype=" << dtype_to_str(a.dtype());
  ss << " place=" << (a.place().is_gpu() ? "gpu" : "cpu");
  ss << ">";
  return ss.str();
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
  m.def("square", [](const Array &x) { return square(x); }, py::arg("x"));
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
  // Signal
  // ====================================================================
  m.def("convolve", &convolve, py::arg("a"), py::arg("v"),
        py::arg("mode") = "full");
  m.def("unwrap", &unwrap, py::arg("p"), py::arg("axis") = -1,
        py::arg("discont") = M_PI, py::arg("period") = 2 * M_PI);
  m.def("sinc", &sinc, py::arg("x"));

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
}
