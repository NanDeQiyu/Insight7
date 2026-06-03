// bindings/lua/insight_lua.cpp
//
// Lua/LuaJIT bindings for the Insight7 scientific computing framework.
// Built with sol2 (header-only C++ <-> Lua binding library).
// API style: PaddlePaddle (ins.float32, ins.CPUPlace(), etc.)
//
// Usage:
//   local ins = require("insight")
//   local a = ins.zeros({2, 3}, ins.float32)
//   local b = ins.ones({2, 3}, ins.float32)
//   local c = a + b
//   print(c)

#define SOL_ALL_SAFETIES_ON 1
#define SOL_NO_EXCEPTIONS 0
#include <sol/sol.hpp>

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
#include "insight/ops/unary.h"
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
#endif

#include <cmath>
#include <csignal>
#include <string>
#include <vector>

using namespace ins;

// Helper: convert Lua table {1,2,3} to std::vector<int64_t>
static std::vector<int64_t> table_to_shape(sol::table t) {
  std::vector<int64_t> v;
  for (size_t i = 1; i <= t.size(); i++) {
    v.push_back(t.get<int64_t>(i));
  }
  return v;
}

// Helper: convert Lua table of Arrays to std::vector<Array>
static std::vector<Array> table_to_arrays(sol::table t) {
  std::vector<Array> v;
  for (size_t i = 1; i <= t.size(); i++) {
    v.push_back(t.get<Array>(i));
  }
  return v;
}

// ── Strict Lua table → Array conversion ──────────────────────────────

// Validate a Lua table element: must be number, boolean, or table.
// nil, string, function, userdata, thread → error.
static void validate_element(sol::object obj, const std::string &path) {
  auto t = obj.get_type();
  if (t == sol::type::lua_nil) {
    throw std::runtime_error("nil found at " + path +
                             ": nil values are not allowed");
  }
  if (t == sol::type::string) {
    throw std::runtime_error("string found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::function) {
    throw std::runtime_error("function found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::userdata || t == sol::type::lightuserdata) {
    throw std::runtime_error("userdata found at " + path +
                             ": only numbers and booleans are allowed");
  }
  if (t == sol::type::thread) {
    throw std::runtime_error("thread found at " + path +
                             ": only numbers and booleans are allowed");
  }
}

// Recursively infer shape from a nested Lua table.
// Throws on ragged nesting or illegal element types.
static void infer_lua_shape(sol::table t, std::vector<int64_t> &shape,
                            int depth, const std::string &path) {
  if (depth >= INSIGHT_MAX_NDIM) {
    throw std::runtime_error(
        "Table nesting exceeds maximum supported dimensions (" +
        std::to_string(INSIGHT_MAX_NDIM) + ") at " + path);
  }
  int64_t len = static_cast<int64_t>(t.size());
  if (depth >= static_cast<int>(shape.size())) {
    shape.push_back(len);
  } else if (shape[depth] != len) {
    throw std::runtime_error(
        "Ragged nested table at " + path + ": dimension " +
        std::to_string(depth) + " has inconsistent sizes (" +
        std::to_string(shape[depth]) + " vs " + std::to_string(len) + ")");
  }
  for (int64_t i = 1; i <= len; i++) {
    sol::object item = t[i];
    std::string child_path = path + "[" + std::to_string(i) + "]";
    validate_element(item, child_path);
    if (item.get_type() == sol::type::table) {
      infer_lua_shape(item.as<sol::table>(), shape, depth + 1, child_path);
    }
  }
}

// Flatten a validated nested Lua table into doubles.
static void flatten_lua_table(sol::table t, std::vector<double> &out) {
  for (size_t i = 1; i <= t.size(); i++) {
    sol::object item = t[i];
    if (item.get_type() == sol::type::table) {
      flatten_lua_table(item.as<sol::table>(), out);
    } else if (item.is<bool>()) {
      out.push_back(item.as<bool>() ? 1.0 : 0.0);
    } else {
      out.push_back(item.as<double>());
    }
  }
}

// Public API: convert a Lua table (nested or flat) to an Array.
// Strict validation: only number/bool/table allowed. nil/string/etc → error.
static Array from_lua_table(sol::table t) {
  if (t.size() == 0) {
    return Array(Shape({0}), DType::F64, CPUPlace());
  }
  std::vector<int64_t> shape;
  std::string path = "root";
  infer_lua_shape(t, shape, 0, path);
  std::vector<double> data;
  flatten_lua_table(t, data);
  int64_t expected = 1;
  for (auto d : shape)
    expected *= d;
  if (static_cast<int64_t>(data.size()) != expected) {
    throw std::runtime_error("Shape/data size mismatch");
  }
  Array result(Shape(shape), DType::F64, CPUPlace());
  std::memcpy(result.data(), data.data(), data.size() * sizeof(double));
  return result;
}

// Helper: Array __tostring — delegates to C++ ins::to_string()
static std::string array_tostring(const Array &a) {
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

// Helper: signal::write_bin wrapper (avoids sol2 default-param issues)
static void lua_write_bin(const std::string &file, const Array &data,
                          sol::optional<bool> append) {
  signal::write_bin(file, data, append.value_or(true));
}

// Helper: signal::read_bin wrapper
static Array lua_read_bin(const std::string &file, sol::optional<int> dtype) {
  DType dt = dtype.has_value() ? static_cast<DType>(dtype.value()) : DType::U8;
  return signal::read_bin(file, dt);
}

// Helper: signal::unpack_bin wrapper (avoids sol2 default-param issues)
static Array lua_unpack_bin(const Array &binary, int dtype,
                            sol::optional<std::string> endianness) {
  return signal::unpack_bin(binary, static_cast<DType>(dtype),
                            endianness.value_or("L"));
}

// Helper: signal::write_sigmf wrapper (avoids sol2 default-param issues)
static void lua_write_sigmf(const std::string &data_file, const Array &data,
                            sol::optional<bool> append) {
  signal::write_sigmf(data_file, data, append.value_or(true));
}

// Convert a Lua 1-based slice spec string to C++ 0-based.
// Positive numbers are decremented by 1; negative numbers and colons unchanged.
// Example: "1:3" → "0:2", "2,1:-1" → "1,0:-1", "::-1" → "::-1"
static std::string lua_spec_to_cpp(const std::string &spec) {
  std::string out;
  for (size_t i = 0; i < spec.size(); i++) {
    char c = spec[i];
    if (c == ':' || c == ',') {
      out += c;
    } else if (c == '-' || (c >= '0' && c <= '9')) {
      size_t start = i;
      if (c == '-')
        i++;
      while (i < spec.size() && spec[i] >= '0' && spec[i] <= '9')
        i++;
      std::string num = spec.substr(start, i - start);
      int val = std::stoi(num);
      if (val > 0) {
        out += std::to_string(val - 1);
      } else {
        out += num;
      }
      i--;
    } else {
      out += c;
    }
  }
  return out;
}

// ============================================================================
// Module entry point
// ============================================================================

extern "C" int luaopen__insight(lua_State *L) {
  sol::state_view lua(L);
  sol::table m = lua.create_table();

  // Auto-initialize CPU backend
  if (!ins::is_initialized()) {
    ins::init({"cpu"});
  }

  // Manual init
  m["init"] = [](sol::table backends) {
    std::vector<std::string> be;
    for (size_t i = 1; i <= backends.size(); i++) {
      be.push_back(backends.get<std::string>(i));
    }
    ins::init(be);
  };

  // Load additional backend after init
  m["load_backend"] = [](const std::string &backend) -> bool {
    try {
      ins::load_backend(backend);
      return true;
    } catch (...) {
      return false;
    }
  };

  // ===== Device information =====
  m["device_name"] = [](sol::optional<std::string> kind,
                        sol::optional<int> device_id) {
    std::string k = kind.value_or("cpu");
    DeviceKind dk =
        (k == "gpu" || k == "cuda") ? DeviceKind::GPU : DeviceKind::CPU;
    return device_name(dk, device_id.value_or(0));
  };
  m["cuda_version"] = []() { return cuda_version(); };
  m["driver_version"] = []() { return driver_version(); };
  m["compute_capability"] = [](sol::optional<int> device_id) {
    return compute_capability(device_id.value_or(0));
  };
  m["device_memory"] = [](sol::optional<int> device_id) {
    auto info = device_memory(device_id.value_or(0));
    return std::make_tuple(info.total, info.free);
  };
  m["gpu_count"] = []() {
    return static_cast<int>(device_count(DeviceKind::GPU));
  };
  m["device_count"] = [](int kind) {
    return static_cast<int>(device_count(static_cast<DeviceKind>(kind)));
  };

  // ===== DType shortcuts (Paddle-style: ins.float32, ins.int64, ...) =====
  m["float32"] = DType::F32;
  m["float64"] = DType::F64;
  m["float16"] = DType::F16;
  m["bfloat16"] = DType::BF16;
  m["int8"] = DType::I8;
  m["int16"] = DType::I16;
  m["int32"] = DType::I32;
  m["int64"] = DType::I64;
  m["uint8"] = DType::U8;
  m["uint16"] = DType::U16;
  m["uint32"] = DType::U32;
  m["uint64"] = DType::U64;
  m["bool"] = DType::BOOL;
  m["complex64"] = DType::C32;
  m["complex128"] = DType::C64;

  // ===== Place constructors =====
  sol::usertype<Place> place_type =
      m.new_usertype<Place>("Place", sol::constructors<Place()>());
  place_type["is_cpu"] = &Place::is_cpu;
  place_type["is_gpu"] = &Place::is_gpu;
  place_type["device_id"] = &Place::device_id;

  m["CPUPlace"] = []() { return CPUPlace(); };
  m["GPUPlace"] = [](int id) { return GPUPlace(id); };

  // ===== Shape helper =====
  m["Shape"] = [](sol::table dims) { return Shape(table_to_shape(dims)); };

  // ===== Array usertype =====
  sol::usertype<Array> array_type = m.new_usertype<Array>(
      "Array",
      sol::constructors<Array(), Array(const Shape &, DType, const Place &)>());

  // Properties
  array_type["shape"] = sol::property([](const Array &a, sol::this_state L) {
    sol::table t = sol::table::create(L.lua_state(), a.shape().ndim());
    for (int i = 0; i < a.shape().ndim(); i++) {
      t[i + 1] = a.shape().dim(i);
    }
    return t;
  });
  array_type["dtype"] = sol::property(&Array::dtype);
  array_type["place"] = sol::property(&Array::place);
  array_type["numel"] = sol::property(&Array::numel);
  array_type["nbytes"] = sol::property(&Array::nbytes);
  array_type["ndim"] =
      sol::property([](const Array &a) { return a.shape().ndim(); });
  array_type["is_contiguous"] = sol::property(&Array::is_contiguous);
  array_type["defined"] = sol::property(&Array::defined);

  // Scalar extraction
  array_type["item"] = [](const Array &a, sol::this_state ts) -> sol::object {
    if (a.numel() != 1) {
      return sol::make_object(ts.lua_state(), sol::lua_nil);
    }
    Array cpu = (a.place().kind() == DeviceKind::CPU) ? a : a.to(CPUPlace());
    double val = 0;
    switch (cpu.dtype()) {
    case DType::F64:
      val = cpu.data<double>()[0];
      break;
    case DType::F32:
      val = (double)cpu.data<float>()[0];
      break;
    case DType::I64:
      val = (double)cpu.data<int64_t>()[0];
      break;
    case DType::I32:
      val = (double)cpu.data<int32_t>()[0];
      break;
    case DType::BOOL:
      val = (double)cpu.data<bool>()[0];
      break;
    default:
      return sol::make_object(ts.lua_state(), sol::lua_nil);
    }
    return sol::make_object(ts.lua_state(), val);
  };
  array_type["get"] = [](const Array &a, int64_t idx) -> double {
    Array cpu = (a.place().kind() == DeviceKind::CPU) ? a : a.to(CPUPlace());
    if (idx < 0 || idx >= cpu.numel())
      return 0.0;
    switch (cpu.dtype()) {
    case DType::F64:
      return cpu.data<double>()[idx];
    case DType::F32:
      return (double)cpu.data<float>()[idx];
    case DType::I64:
      return (double)cpu.data<int64_t>()[idx];
    case DType::I32:
      return (double)cpu.data<int32_t>()[idx];
    default:
      return 0.0;
    }
  };

  // View ops
  array_type["contiguous"] = &Array::contiguous;
  array_type["reshape"] = [](const Array &a, sol::table new_shape) {
    return a.reshape(Shape(table_to_shape(new_shape)));
  };
  array_type["transpose"] = sol::overload(
      [](const Array &a) { return a.transpose(); },
      [](const Array &a, std::vector<int> perm) { return a.transpose(perm); });
  array_type["squeeze"] = [](const Array &a, sol::optional<int> axis) {
    return a.squeeze(axis ? std::optional<int>(*axis) : std::nullopt);
  };
  array_type["unsqueeze"] = &Array::unsqueeze;

  // Device/type conversion
  array_type["to"] = sol::overload(
      [](const Array &a, const Place &p) { return a.to(p); },
      [](const Array &a, DType dt) { return a.to(dt); },
      [](const Array &a, const Place &p, DType dt) { return a.to(p, dt); });
  array_type["copy"] = &Array::copy;

  // String slicing: a[":,1:-1"] (Lua 1-based → auto-convert to 0-based)
  // Integer indexing: a[1] → first element, a[3] → third element
  array_type[sol::meta_function::index] = sol::overload(
      [](const Array &a, const std::string &spec) {
        return a[lua_spec_to_cpp(spec)];
      },
      [](const Array &a, int64_t idx) {
        // 1-based integer indexing → 0-based
        if (idx == 0)
          throw std::runtime_error(
              "Lua arrays are 1-based: index 0 is invalid");
        if (idx > 0)
          idx -= 1;
        return a.at({idx});
      },
      [](const Array &a, const Slice &s) { return a[s]; });

  // Arithmetic metamethods
  array_type[sol::meta_function::addition] =
      sol::overload([](const Array &a, const Array &b) { return a + b; },
                    [](const Array &a, double b) { return a + b; },
                    [](double a, const Array &b) { return a + b; });
  array_type[sol::meta_function::subtraction] =
      sol::overload([](const Array &a, const Array &b) { return a - b; },
                    [](const Array &a, double b) { return a - b; },
                    [](double a, const Array &b) { return a - b; });
  array_type[sol::meta_function::multiplication] =
      sol::overload([](const Array &a, const Array &b) { return a * b; },
                    [](const Array &a, double b) { return a * b; },
                    [](double a, const Array &b) { return a * b; });
  array_type[sol::meta_function::division] =
      sol::overload([](const Array &a, const Array &b) { return a / b; },
                    [](const Array &a, double b) { return a / b; },
                    [](double a, const Array &b) { return a / b; });
  array_type[sol::meta_function::modulus] =
      sol::overload([](const Array &a, const Array &b) { return a % b; },
                    [](const Array &a, double b) { return a % b; });
  array_type[sol::meta_function::unary_minus] = [](const Array &a) {
    return -a;
  };
  array_type[sol::meta_function::bitwise_not] = [](const Array &a) {
    return ~a;
  };

  // Comparison metamethods
  array_type[sol::meta_function::equal_to] =
      [](const Array &a, const Array &b) { return equal(a, b); };
  array_type[sol::meta_function::less_than] =
      [](const Array &a, const Array &b) { return less(a, b); };
  array_type[sol::meta_function::less_than_or_equal_to] =
      [](const Array &a, const Array &b) { return less_equal(a, b); };

  // String representation metamethod (enables print(arr) and tostring(arr))
  array_type[sol::meta_function::to_string] = &array_tostring;

  // ====================================================================
  // Creation
  // ====================================================================
  m["from_table"] = &from_lua_table;
  m["zeros"] = [](sol::table shape, sol::optional<DType> dtype,
                  sol::optional<Place> place) {
    return zeros(Shape(table_to_shape(shape)), dtype.value_or(DType::F32),
                 place.value_or(get_device()));
  };
  m["ones"] = [](sol::table shape, sol::optional<DType> dtype,
                 sol::optional<Place> place) {
    return ones(Shape(table_to_shape(shape)), dtype.value_or(DType::F32),
                place.value_or(get_device()));
  };
  m["full"] = [](sol::table shape, double fill, sol::optional<DType> dtype,
                 sol::optional<Place> place) {
    return full(Shape(table_to_shape(shape)), fill, dtype.value_or(DType::F32),
                place.value_or(get_device()));
  };
  m["eye"] = [](int64_t n, sol::optional<int64_t> m_opt,
                sol::optional<int64_t> k, sol::optional<DType> dtype,
                sol::optional<Place> place) {
    return eye(n, m_opt.value_or(-1), k.value_or(0), dtype.value_or(DType::F32),
               place.value_or(get_device()));
  };
  m["arange"] = sol::overload(
      [](double end, sol::optional<DType> dtype, sol::optional<Place> place) {
        return arange(end, dtype.value_or(DType::I64),
                      place.value_or(get_device()));
      },
      [](double start, double end, sol::optional<double> step,
         sol::optional<DType> dtype, sol::optional<Place> place) {
        return arange(start, end, step.value_or(1.0),
                      dtype.value_or(DType::I64), place.value_or(get_device()));
      });
  m["linspace"] = [](double start, double stop, int64_t num,
                     sol::optional<DType> dtype, sol::optional<Place> place) {
    return linspace(start, stop, num, dtype.value_or(DType::F32),
                    place.value_or(get_device()));
  };
  m["zeros_like"] = &zeros_like;
  m["ones_like"] = &ones_like;

  // ====================================================================
  // Elementwise
  // ====================================================================
  m["add"] = [](const Array &a, const Array &b) { return add(a, b); };
  m["sub"] = [](const Array &a, const Array &b) { return sub(a, b); };
  m["mul"] = [](const Array &a, const Array &b) { return mul(a, b); };
  m["div"] = [](const Array &a, const Array &b) { return div(a, b); };
  m["pow"] = [](const Array &a, const Array &b) { return pow(a, b); };
  m["mod"] = [](const Array &a, const Array &b) { return mod(a, b); };
  m["maximum"] = [](const Array &a, const Array &b) { return maximum(a, b); };
  m["minimum"] = [](const Array &a, const Array &b) { return minimum(a, b); };
  m["equal"] = [](const Array &a, const Array &b) { return equal(a, b); };
  m["not_equal"] = [](const Array &a, const Array &b) {
    return not_equal(a, b);
  };
  m["greater"] = [](const Array &a, const Array &b) { return greater(a, b); };
  m["less"] = [](const Array &a, const Array &b) { return less(a, b); };
  m["greater_equal"] = [](const Array &a, const Array &b) {
    return greater_equal(a, b);
  };
  m["less_equal"] = [](const Array &a, const Array &b) {
    return less_equal(a, b);
  };
  m["logical_and"] = [](const Array &a, const Array &b) {
    return logical_and(a, b);
  };
  m["logical_or"] = [](const Array &a, const Array &b) {
    return logical_or(a, b);
  };
  m["logical_xor"] = [](const Array &a, const Array &b) {
    return logical_xor(a, b);
  };
  m["logical_not"] = [](const Array &x) { return logical_not(x); };

  // ====================================================================
  // Unary math
  // ====================================================================
  m["abs"] = [](const Array &x) { return ins::abs(x); };
  m["negative"] = [](const Array &x) { return negative(x); };
  m["square"] = [](const Array &x) { return square(x); };
  m["sqrt"] = [](const Array &x) { return sqrt(x); };
  m["exp"] = [](const Array &x) { return exp(x); };
  m["log"] = [](const Array &x) { return log(x); };
  m["log2"] = [](const Array &x) { return log2(x); };
  m["log10"] = [](const Array &x) { return log10(x); };
  m["sin"] = [](const Array &x) { return sin(x); };
  m["cos"] = [](const Array &x) { return cos(x); };
  m["tan"] = [](const Array &x) { return tan(x); };
  m["asin"] = [](const Array &x) { return asin(x); };
  m["acos"] = [](const Array &x) { return acos(x); };
  m["atan"] = [](const Array &x) { return atan(x); };
  m["sinh"] = [](const Array &x) { return sinh(x); };
  m["cosh"] = [](const Array &x) { return cosh(x); };
  m["tanh"] = [](const Array &x) { return tanh(x); };
  m["floor"] = [](const Array &x) { return floor(x); };
  m["ceil"] = [](const Array &x) { return ceil(x); };
  m["round"] = [](const Array &x) { return rint(x); };
  m["sign"] = [](const Array &x) { return sign(x); };
  m["conj"] = &conj;
  m["angle"] = &angle;
  m["isnan"] = [](const Array &x) { return isnan(x); };
  m["isinf"] = [](const Array &x) { return isinf(x); };
  m["isfinite"] = [](const Array &x) { return isfinite(x); };
  m["where"] = [](const Array &cond, const Array &x, const Array &y) {
    return where(cond, x, y);
  };
  m["exp2"] = [](const Array &x) { return exp2(x); };
  m["expm1"] = [](const Array &x) { return expm1(x); };
  m["log1p"] = [](const Array &x) { return log1p(x); };
  m["cbrt"] = [](const Array &x) { return cbrt(x); };
  m["reciprocal"] = [](const Array &x) { return reciprocal(x); };
  m["asinh"] = [](const Array &x) { return asinh(x); };
  m["acosh"] = [](const Array &x) { return acosh(x); };
  m["atanh"] = [](const Array &x) { return atanh(x); };
  m["trunc"] = [](const Array &x) { return trunc(x); };
  m["deg2rad"] = [](const Array &x) { return deg2rad(x); };
  m["rad2deg"] = [](const Array &x) { return rad2deg(x); };

  // ====================================================================
  // Complex
  // ====================================================================
  m["is_complex"] = [](const Array &x) { return is_complex(x); };
  m["has_complex_shape"] = [](const Array &x) { return has_complex_shape(x); };
  m["to_complex"] =
      sol::overload([](const Array &real) { return to_complex(real); },
                    [](const Array &real, const Array &imag) {
                      return to_complex(real, imag);
                    });
  m["as_complex"] = [](const Array &x) { return as_complex(x); };
  m["as_real"] = [](const Array &x) { return as_real(x); };
  m["real"] = [](const Array &z) { return real(z); };
  m["imag"] = [](const Array &z) { return imag(z); };

  // ====================================================================
  // Reduction
  // ====================================================================
  m["sum"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return sum(x, ax, keepdims.value_or(false));
  };
  m["mean"] = [](const Array &x, sol::optional<int> axis,
                 sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return mean(x, ax, keepdims.value_or(false));
  };
  m["max"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return max(x, ax, keepdims.value_or(false));
  };
  m["min"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return min(x, ax, keepdims.value_or(false));
  };
  m["prod"] = [](const Array &x, sol::optional<int> axis,
                 sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return prod(x, ax, keepdims.value_or(false));
  };
  m["argmax"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return argmax(x, ax, keepdims.value_or(false));
  };
  m["argmin"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return argmin(x, ax, keepdims.value_or(false));
  };
  m["any"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return any(x, ax, keepdims.value_or(false));
  };
  m["all"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return all(x, ax, keepdims.value_or(false));
  };
  m["var"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return var(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["std"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return ins::std(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["cumsum"] = [](const Array &x, int axis) { return cumsum(x, axis); };
  m["cumprod"] = [](const Array &x, int axis) { return cumprod(x, axis); };
  m["cummax"] = [](const Array &x, int axis) { return cummax(x, axis); };
  m["cummin"] = [](const Array &x, int axis) { return cummin(x, axis); };
  m["sem"] = [](const Array &x, sol::optional<int> axis,
                sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return sem(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["count_nonzero"] = [](const Array &x, sol::optional<int> axis,
                          sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return count_nonzero(x, ax, keepdims.value_or(false));
  };
  m["median"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return median(x, ax, keepdims.value_or(false));
  };
  m["quantile"] = sol::overload(
      [](const Array &x, double q, sol::optional<int> axis,
         sol::optional<bool> keepdims) {
        std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
        return quantile(x, q, ax, keepdims.value_or(false));
      },
      [](const Array &x, const Array &q, sol::optional<int> axis,
         sol::optional<bool> keepdims) {
        std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
        return quantile(x, q, ax, keepdims.value_or(false));
      });
  m["percentile"] = [](const Array &x, double q, sol::optional<int> axis,
                       sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return percentile(x, q, ax, keepdims.value_or(false));
  };
  m["nansum"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nansum(x, ax, keepdims.value_or(false));
  };
  m["nanmean"] = [](const Array &x, sol::optional<int> axis,
                    sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nanmean(x, ax, keepdims.value_or(false));
  };
  m["nanmax"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nanmax(x, ax, keepdims.value_or(false));
  };
  m["nanmin"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nanmin(x, ax, keepdims.value_or(false));
  };
  m["nanstd"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nanstd(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };
  m["nanvar"] = [](const Array &x, sol::optional<int> axis,
                   sol::optional<bool> keepdims, sol::optional<int> ddof) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return nanvar(x, ax, keepdims.value_or(false), ddof.value_or(0));
  };

  // ====================================================================
  // Manipulation
  // ====================================================================
  m["concat"] = [](sol::table arrays, sol::optional<int> axis) {
    return concat(table_to_arrays(arrays), axis.value_or(0));
  };
  m["stack"] = [](sol::table arrays, sol::optional<int> axis) {
    return stack(table_to_arrays(arrays), axis.value_or(0));
  };
  m["split"] = [](const Array &x, int sections, sol::optional<int> axis) {
    return split(x, sections, axis.value_or(0));
  };
  m["tile"] = [](const Array &x, const Shape &reps) { return tile(x, reps); };
  m["repeat"] = [](const Array &x, int repeats, sol::optional<int> axis) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return repeat(x, repeats, ax);
  };
  m["flip"] = [](const Array &x, sol::optional<int> axis) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return flip(x, ax);
  };
  m["pad"] = [](const Array &x, std::vector<int64_t> pad_width,
                sol::optional<double> val) {
    return pad(x, pad_width, val.value_or(0.0));
  };
  m["flatten"] = [](const Array &x) { return flatten(x); };
  m["ravel"] = [](const Array &x) { return ravel(x); };
  m["roll"] = [](const Array &x, int shift, sol::optional<int> axis) {
    std::optional<int> ax = axis ? std::optional<int>(*axis) : std::nullopt;
    return roll(x, shift, ax);
  };
  m["permute"] = [](const Array &x, std::vector<int> axes) {
    return permute(x, axes);
  };
  m["swapaxes"] = [](const Array &x, int axis1, int axis2) {
    return swapaxes(x, axis1, axis2);
  };
  m["moveaxis"] = [](const Array &x, int source, int destination) {
    return moveaxis(x, source, destination);
  };
  m["fliplr"] = [](const Array &x) { return fliplr(x); };
  m["flipud"] = [](const Array &x) { return flipud(x); };
  m["rot90"] = [](const Array &x, sol::optional<int> k,
                  sol::optional<std::vector<int>> axes) {
    return rot90(x, k.value_or(1), axes.value_or(std::vector<int>{0, 1}));
  };
  m["diag"] = [](const Array &x, sol::optional<int> k) {
    return diag(x, k.value_or(0));
  };
  m["diagonal"] = [](const Array &x, sol::optional<int> offset,
                     sol::optional<int> axis1, sol::optional<int> axis2) {
    return diagonal(x, offset.value_or(0), axis1.value_or(0),
                    axis2.value_or(1));
  };
  m["tril"] = [](const Array &x, sol::optional<int> k) {
    return tril(x, k.value_or(0));
  };
  m["triu"] = [](const Array &x, sol::optional<int> k) {
    return triu(x, k.value_or(0));
  };
  m["diff"] = [](const Array &x, sol::optional<int> n,
                 sol::optional<int> axis) {
    return diff(x, n.value_or(1), axis.value_or(-1));
  };

  // ====================================================================
  // Linear Algebra
  // ====================================================================
  m["matmul"] = [](const Array &a, const Array &b) { return matmul(a, b); };
  m["dot"] = [](const Array &a, const Array &b) { return dot(a, b); };
  m["outer"] = [](const Array &a, const Array &b) { return outer(a, b); };
  m["det"] = [](const Array &x) { return det(x); };
  m["inv"] = [](const Array &x) { return inv(x); };
  m["solve"] = [](const Array &a, const Array &b) { return solve(a, b); };
  m["svd"] = [](const Array &x) { return svd(x); };
  m["eigh"] = [](const Array &x) { return eigh(x); };
  m["cholesky"] = [](const Array &x, sol::optional<bool> lower) {
    return cholesky(x, lower.value_or(true));
  };
  m["qr"] = [](const Array &x) { return qr(x); };
  m["norm"] = [](const Array &x, sol::optional<double> ord) {
    return ins::norm(x, ord.value_or(2.0));
  };
  m["trace"] = [](const Array &x) { return trace(x); };
  m["lstsq"] = [](const Array &a, const Array &b) { return lstsq(a, b); };
  m["cond"] = [](const Array &x, sol::optional<double> p) {
    return cond(x, p.value_or(2.0));
  };
  m["matrix_rank"] = [](const Array &x) { return matrix_rank(x); };
  m["matrix_power"] = [](const Array &x, int n) { return matrix_power(x, n); };
  m["slogdet"] = [](const Array &x) { return slogdet(x); };
  m["eigvalsh"] = [](const Array &x, sol::optional<std::string> uplo) {
    return eigvalsh(x, uplo.value_or("L"));
  };

  // ====================================================================
  // FFT
  // ====================================================================
  m["fft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                sol::optional<std::string> norm) {
    return fft::fft(x, n.value_or(-1), axis.value_or(-1),
                    norm.value_or("backward"));
  };
  m["ifft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                 sol::optional<std::string> norm) {
    return fft::ifft(x, n.value_or(-1), axis.value_or(-1),
                     norm.value_or("backward"));
  };
  m["rfft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                 sol::optional<std::string> norm) {
    return fft::rfft(x, n.value_or(-1), axis.value_or(-1),
                     norm.value_or("backward"));
  };
  m["irfft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                  sol::optional<std::string> norm) {
    return fft::irfft(x, n.value_or(-1), axis.value_or(-1),
                      norm.value_or("backward"));
  };
  m["fftfreq"] = [](int64_t n, sol::optional<double> d) {
    return fft::fftfreq(n, d.value_or(1.0));
  };
  m["rfftfreq"] = [](int64_t n, sol::optional<double> d) {
    return fft::rfftfreq(n, d.value_or(1.0));
  };
  m["fftshift"] = [](const Array &x, sol::optional<int> axis) {
    return fft::fftshift(x, axis.value_or(-1));
  };
  m["ifftshift"] = [](const Array &x, sol::optional<int> axis) {
    return fft::ifftshift(x, axis.value_or(-1));
  };
  m["next_fast_len"] = [](int target) { return fft::next_fast_len(target); };
  m["hfft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                 sol::optional<std::string> norm) {
    return fft::hfft(x, n.value_or(-1), axis.value_or(-1),
                     norm.value_or("backward"));
  };
  m["ihfft"] = [](const Array &x, sol::optional<int> n, sol::optional<int> axis,
                  sol::optional<std::string> norm) {
    return fft::ihfft(x, n.value_or(-1), axis.value_or(-1),
                      norm.value_or("backward"));
  };
  m["rfft2"] = [](const Array &x, sol::optional<std::vector<int64_t>> s,
                  sol::optional<std::vector<int>> axes,
                  sol::optional<std::string> norm) {
    return fft::rfft2(x, s.value_or(std::vector<int64_t>{}),
                      axes.value_or(std::vector<int>{-2, -1}),
                      norm.value_or("backward"));
  };
  m["irfft2"] = [](const Array &x, sol::optional<std::vector<int64_t>> s,
                   sol::optional<std::vector<int>> axes,
                   sol::optional<std::string> norm) {
    return fft::irfft2(x, s.value_or(std::vector<int64_t>{}),
                       axes.value_or(std::vector<int>{-2, -1}),
                       norm.value_or("backward"));
  };
  m["fft2"] = [](const Array &x, sol::optional<sol::table> s,
                 sol::optional<sol::table> axes,
                 sol::optional<std::string> norm) {
    std::vector<int64_t> s_vec;
    if (s) {
      for (auto &kv : *s)
        s_vec.push_back(kv.second.as<int64_t>());
    }
    std::vector<int> axes_vec = {-2, -1};
    if (axes) {
      axes_vec.clear();
      for (auto &kv : *axes)
        axes_vec.push_back(kv.second.as<int>());
    }
    return fft::fft2(x, s_vec, axes_vec, norm.value_or("backward"));
  };
  m["ifft2"] = [](const Array &x, sol::optional<sol::table> s,
                  sol::optional<sol::table> axes,
                  sol::optional<std::string> norm) {
    std::vector<int64_t> s_vec;
    if (s) {
      for (auto &kv : *s)
        s_vec.push_back(kv.second.as<int64_t>());
    }
    std::vector<int> axes_vec = {-2, -1};
    if (axes) {
      axes_vec.clear();
      for (auto &kv : *axes)
        axes_vec.push_back(kv.second.as<int>());
    }
    return fft::ifft2(x, s_vec, axes_vec, norm.value_or("backward"));
  };
  m["rfftn"] = [](const Array &x, sol::optional<std::vector<int64_t>> s,
                  sol::optional<std::vector<int>> axes,
                  sol::optional<std::string> norm) {
    return fft::rfftn(x, s.value_or(std::vector<int64_t>{}),
                      axes.value_or(std::vector<int>{}),
                      norm.value_or("backward"));
  };
  m["irfftn"] = [](const Array &x, sol::optional<std::vector<int64_t>> s,
                   sol::optional<std::vector<int>> axes,
                   sol::optional<std::string> norm) {
    return fft::irfftn(x, s.value_or(std::vector<int64_t>{}),
                       axes.value_or(std::vector<int>{}),
                       norm.value_or("backward"));
  };

  // ====================================================================
  // Signal (ins.signal.*)
  // ====================================================================
  {
    sol::table sig = m.create_named("signal");

    // --- ChirpMethod enum ---
    sol::table cm = sig.create_named("ChirpMethod");
    cm["linear"] = signal::ChirpMethod::Linear;
    cm["quadratic"] = signal::ChirpMethod::Quadratic;
    cm["logarithmic"] = signal::ChirpMethod::Logarithmic;
    cm["hyperbolic"] = signal::ChirpMethod::Hyperbolic;

    // --- Windows ---
    sig["general_cosine"] = [](int64_t M, std::vector<double> a,
                               sol::optional<bool> sym) {
      return signal::general_cosine(M, a, sym.value_or(true));
    };
    sig["get_window"] = sol::overload(
        [](const std::string &w, int64_t Nx, sol::optional<bool> fftbins) {
          return signal::get_window(w, Nx, fftbins.value_or(true));
        },
        [](const std::string &w, double param, int64_t Nx,
           sol::optional<bool> fftbins) {
          return signal::get_window(w, param, Nx, fftbins.value_or(true));
        });
    sig["boxcar"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::boxcar(M, sym.value_or(true));
    };
    sig["triang"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::triang(M, sym.value_or(true));
    };
    sig["parzen"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::parzen(M, sym.value_or(true));
    };
    sig["bohman"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::bohman(M, sym.value_or(true));
    };
    sig["bartlett"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::bartlett(M, sym.value_or(true));
    };
    sig["cosine"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::cosine(M, sym.value_or(true));
    };
    sig["exponential"] = [](int64_t M, sol::optional<double> center,
                            sol::optional<double> tau,
                            sol::optional<bool> sym) {
      return signal::exponential(M, center.value_or(-1.0), tau.value_or(1.0),
                                 sym.value_or(true));
    };
    sig["blackman"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::blackman(M, sym.value_or(true));
    };
    sig["nuttall"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::nuttall(M, sym.value_or(true));
    };
    sig["blackmanharris"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::blackmanharris(M, sym.value_or(true));
    };
    sig["flattop"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::flattop(M, sym.value_or(true));
    };
    sig["hann"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::hann(M, sym.value_or(true));
    };
    sig["general_hamming"] = [](int64_t M, double alpha,
                                sol::optional<bool> sym) {
      return signal::general_hamming(M, alpha, sym.value_or(true));
    };
    sig["hamming"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::hamming(M, sym.value_or(true));
    };
    sig["tukey"] = [](int64_t M, sol::optional<double> alpha,
                      sol::optional<bool> sym) {
      return signal::tukey(M, alpha.value_or(0.5), sym.value_or(true));
    };
    sig["barthann"] = [](int64_t M, sol::optional<bool> sym) {
      return signal::barthann(M, sym.value_or(true));
    };
    sig["kaiser"] = [](int64_t M, double beta, sol::optional<bool> sym) {
      return signal::kaiser(M, beta, sym.value_or(true));
    };
    sig["gaussian"] = [](int64_t M, double std_val, sol::optional<bool> sym) {
      return signal::gaussian(M, std_val, sym.value_or(true));
    };
    sig["general_gaussian"] = [](int64_t M, double p, double sig_val,
                                 sol::optional<bool> sym) {
      return signal::general_gaussian(M, p, sig_val, sym.value_or(true));
    };
    sig["chebwin"] = [](int64_t M, double at, sol::optional<bool> sym) {
      return signal::chebwin(M, at, sym.value_or(true));
    };
    sig["taylor"] = [](int64_t M, sol::optional<int64_t> nbar,
                       sol::optional<double> sll, sol::optional<bool> norm,
                       sol::optional<bool> sym) {
      return signal::taylor(M, nbar.value_or(4), sll.value_or(-30.0),
                            norm.value_or(true), sym.value_or(true));
    };

    // --- Waveforms ---
    sig["sawtooth"] = [](const Array &t, sol::optional<double> width) {
      return signal::sawtooth(t, width.value_or(1.0));
    };
    sig["square_wf"] = [](const Array &t, sol::optional<double> duty) {
      return signal::square(t, duty.value_or(0.5));
    };
    sig["square"] = sig["square_wf"]; // alias for signal::square wave
    sig["gausspulse"] = [](const Array &t, sol::optional<double> fc,
                           sol::optional<double> bw, sol::optional<double> bwr,
                           sol::optional<double> tpr) {
      return signal::gausspulse(t, fc.value_or(1000), bw.value_or(0.5),
                                bwr.value_or(-6), tpr.value_or(-60));
    };
    sig["chirp"] = [](const Array &t, double f0, double t1, double f1,
                      sol::optional<signal::ChirpMethod> method,
                      sol::optional<double> phi,
                      sol::optional<bool> vertex_zero) {
      return signal::chirp(t, f0, t1, f1,
                           method.value_or(signal::ChirpMethod::Linear),
                           phi.value_or(0.0), vertex_zero.value_or(true));
    };
    sig["unit_impulse"] = [](sol::table shape, sol::optional<int64_t> idx,
                             sol::optional<DType> dtype,
                             sol::optional<Place> place) {
      return signal::unit_impulse(Shape(table_to_shape(shape)),
                                  idx.value_or(-1), dtype.value_or(DType::F64),
                                  place.value_or(get_device()));
    };

    // --- B-Splines ---
    sig["gauss_spline"] = &signal::gauss_spline;
    sig["cubic"] = &signal::cubic;
    sig["quadratic"] = &signal::quadratic;

    // --- Filter Design ---
    sig["kaiser_beta"] = &signal::kaiser_beta;
    sig["kaiser_atten"] = &signal::kaiser_atten;
    sig["firwin"] = [](int64_t numtaps, std::vector<double> cutoff,
                       sol::optional<std::string> window,
                       sol::optional<std::string> pass_zero,
                       sol::optional<bool> scale) {
      return signal::firwin(numtaps, cutoff, window.value_or("hamming"),
                            pass_zero.value_or("lowpass"),
                            scale.value_or(true));
    };
    sig["firwin2"] = [](int64_t numtaps, std::vector<double> freq,
                        std::vector<double> gain, sol::optional<int64_t> nfreqs,
                        sol::optional<std::string> window,
                        sol::optional<bool> antisymmetric) {
      return signal::firwin2(numtaps, freq, gain, nfreqs.value_or(0),
                             window.value_or("hamming"),
                             antisymmetric.value_or(false));
    };
    sig["cmplx_sort"] = &signal::cmplx_sort;

    // --- Convolution ---
    sig["fftconvolve"] = [](const Array &in1, const Array &in2,
                            sol::optional<std::string> mode) {
      return signal::fftconvolve(in1, in2, mode.value_or("full"));
    };
    sig["correlate"] = [](const Array &in1, const Array &in2,
                          sol::optional<std::string> mode) {
      return signal::correlate(in1, in2, mode.value_or("full"));
    };
    sig["convolve2d"] = [](const Array &in1, const Array &in2,
                           sol::optional<std::string> mode) {
      return signal::convolve2d(in1, in2, mode.value_or("full"));
    };
    sig["correlate2d"] = [](const Array &in1, const Array &in2,
                            sol::optional<std::string> mode) {
      return signal::correlate2d(in1, in2, mode.value_or("full"));
    };
    sig["choose_conv_method"] = [](const Array &in1, const Array &in2,
                                   sol::optional<std::string> mode) {
      return signal::choose_conv_method(in1, in2, mode.value_or("full"));
    };
    sig["correlation_lags"] = [](int64_t in1_len, int64_t in2_len,
                                 sol::optional<std::string> mode) {
      return signal::correlation_lags(in1_len, in2_len, mode.value_or("full"));
    };

    // --- Filtering ---
    sig["hilbert"] = [](const Array &x, sol::optional<int64_t> N) {
      return signal::hilbert(x, N.value_or(-1));
    };
    sig["hilbert2"] = [](const Array &x, sol::optional<int64_t> N) {
      return signal::hilbert2(x, N.value_or(-1));
    };
    sig["detrend"] = [](const Array &data, sol::optional<int> axis,
                        sol::optional<std::string> type) {
      return signal::detrend(data, axis.value_or(-1), type.value_or("linear"));
    };
    sig["wiener"] = [](const Array &im,
                       sol::optional<std::vector<int64_t>> mysize,
                       sol::optional<double> noise) {
      return signal::wiener(im, mysize.value_or(std::vector<int64_t>{}),
                            noise.value_or(-1.0));
    };
    sig["firfilter"] = [](const Array &b, const Array &x,
                          sol::optional<int> axis) {
      return signal::firfilter(b, x, axis.value_or(-1));
    };
    sig["lfilter"] = [](const Array &b, const Array &a, const Array &x,
                        sol::optional<int> axis) {
      return signal::lfilter(b, a, x, axis.value_or(-1));
    };
    sig["lfilter_zi"] = &signal::lfilter_zi;
    sig["filtfilt"] = [](const Array &b, const Array &a, const Array &x,
                         sol::optional<int> axis) {
      return signal::filtfilt(b, a, x, axis.value_or(-1));
    };
    sig["decimate"] = [](const Array &x, int64_t q, sol::optional<int> axis,
                         sol::optional<bool> zero_phase) {
      return signal::decimate(x, q, axis.value_or(-1),
                              zero_phase.value_or(true));
    };
    sig["resample"] = [](const Array &x, int64_t num, sol::optional<int> axis) {
      return signal::resample(x, num, axis.value_or(-1));
    };
    sig["resample_poly"] = [](const Array &x, int64_t up, int64_t down,
                              sol::optional<int> axis) {
      return signal::resample_poly(x, up, down, axis.value_or(-1));
    };
    sig["freq_shift"] = &signal::freq_shift;

    // --- Spectral Analysis ---
    sig["welch"] =
        [](const Array &x, sol::optional<double> fs,
           sol::optional<std::string> window, sol::optional<int64_t> nperseg,
           sol::optional<int64_t> noverlap, sol::optional<int64_t> nfft,
           sol::optional<std::string> detrend,
           sol::optional<bool> return_onesided,
           sol::optional<std::string> scaling) {
          return signal::welch(x, fs.value_or(1.0), window.value_or("hann"),
                               nperseg.value_or(256), noverlap.value_or(0),
                               nfft.value_or(0), detrend.value_or("constant"),
                               return_onesided.value_or(true),
                               scaling.value_or("density"));
        };
    sig["periodogram"] = [](const Array &x, sol::optional<double> fs,
                            sol::optional<std::string> window,
                            sol::optional<int64_t> nfft,
                            sol::optional<std::string> detrend,
                            sol::optional<bool> return_onesided,
                            sol::optional<std::string> scaling) {
      return signal::periodogram(x, fs.value_or(1.0), window.value_or("boxcar"),
                                 nfft.value_or(0), detrend.value_or("constant"),
                                 return_onesided.value_or(true),
                                 scaling.value_or("density"));
    };
    sig["csd"] =
        [](const Array &x, const Array &y, sol::optional<double> fs,
           sol::optional<std::string> window, sol::optional<int64_t> nperseg,
           sol::optional<int64_t> noverlap, sol::optional<int64_t> nfft,
           sol::optional<std::string> detrend,
           sol::optional<bool> return_onesided,
           sol::optional<std::string> scaling) {
          return signal::csd(x, y, fs.value_or(1.0), window.value_or("hann"),
                             nperseg.value_or(256), noverlap.value_or(0),
                             nfft.value_or(0), detrend.value_or("constant"),
                             return_onesided.value_or(true),
                             scaling.value_or("density"));
        };
    sig["coherence"] = [](const Array &x, const Array &y,
                          sol::optional<double> fs,
                          sol::optional<std::string> window,
                          sol::optional<int64_t> nperseg,
                          sol::optional<int64_t> noverlap,
                          sol::optional<int64_t> nfft,
                          sol::optional<std::string> detrend) {
      return signal::coherence(x, y, fs.value_or(1.0), window.value_or("hann"),
                               nperseg.value_or(256), noverlap.value_or(0),
                               nfft.value_or(0), detrend.value_or("constant"));
    };
    sig["spectrogram"] = [](const Array &x, sol::optional<double> fs,
                            sol::optional<std::string> window,
                            sol::optional<int64_t> nperseg,
                            sol::optional<int64_t> noverlap,
                            sol::optional<int64_t> nfft,
                            sol::optional<std::string> detrend,
                            sol::optional<bool> return_onesided,
                            sol::optional<std::string> mode) {
      return signal::spectrogram(
          x, fs.value_or(1.0), window.value_or("hann"), nperseg.value_or(256),
          noverlap.value_or(0), nfft.value_or(0), detrend.value_or("constant"),
          return_onesided.value_or(true), mode.value_or("psd"));
    };
    sig["stft"] =
        [](const Array &x, sol::optional<double> fs,
           sol::optional<std::string> window, sol::optional<int64_t> nperseg,
           sol::optional<int64_t> noverlap, sol::optional<int64_t> nfft) {
          return signal::stft(x, fs.value_or(1.0), window.value_or("hann"),
                              nperseg.value_or(256), noverlap.value_or(0),
                              nfft.value_or(0));
        };
    sig["vectorstrength"] = &signal::vectorstrength;

    // --- Wavelets ---
    sig["morlet"] = [](int64_t M, sol::optional<double> w,
                       sol::optional<double> s, sol::optional<bool> complete) {
      return signal::morlet(M, w.value_or(5.0), s.value_or(1.0),
                            complete.value_or(true));
    };
    sig["ricker"] = &signal::ricker;
    sig["morlet2"] = [](int64_t M, double s, sol::optional<double> w) {
      return signal::morlet2(M, s, w.value_or(5.0));
    };
    // Note: cwt takes std::function — not easily bindable from Lua

    // --- Acoustics ---
    sig["mel2hz"] = &signal::mel2hz;
    sig["hz2mel"] = &signal::hz2mel;
    sig["mel_frequencies"] = [](int64_t n_mels, sol::optional<double> f_low,
                                sol::optional<double> f_high) {
      return signal::mel_frequencies(n_mels, f_low.value_or(0.0),
                                     f_high.value_or(11000.0));
    };
    sig["hz2bark"] = &signal::hz2bark;
    sig["bark2hz"] = &signal::bark2hz;

    // --- Demod ---
    sig["fm_demod"] = [](const Array &x, sol::optional<int> axis) {
      return signal::fm_demod(x, axis.value_or(-1));
    };

    // --- Peak Finding ---
    sig["argrelextrema"] = &signal::argrelextrema;
    sig["argrelmax"] = &signal::argrelmax;
    sig["argrelmin"] = &signal::argrelmin;

    // --- Radar ---
    sig["pulse_compression"] = [](const Array &x, const Array &tpl,
                                  sol::optional<bool> normalize,
                                  sol::optional<std::string> window,
                                  sol::optional<int64_t> nfft) {
      return signal::pulse_compression(x, tpl, normalize.value_or(false),
                                       window.value_or(""), nfft.value_or(0));
    };
    sig["pulse_doppler"] = &signal::pulse_doppler;
    sig["cfar_alpha"] = &signal::cfar_alpha;
    sig["ca_cfar"] = &signal::ca_cfar;
    sig["mvdr"] = [](const Array &x, const Array &sv,
                     sol::optional<bool> calc_cov) {
      return signal::mvdr(x, sv, calc_cov.value_or(true));
    };
    sig["ambgfun"] = &signal::ambgfun;

    // --- Signal I/O ---
    sig["read_bin"] = &lua_read_bin;
    sig["write_bin"] = &lua_write_bin;
    sig["unpack_bin"] = &lua_unpack_bin;
    sig["pack_bin"] = &signal::pack_bin;
    sig["read_sigmf"] = &signal::read_sigmf;
    sig["write_sigmf"] = &lua_write_sigmf;

    // --- Estimation (KalmanFilter) ---
    sol::usertype<signal::KalmanFilter> kf_type =
        sig.new_usertype<signal::KalmanFilter>("KalmanFilter");
    kf_type["x"] = &signal::KalmanFilter::x;
    kf_type["P"] = &signal::KalmanFilter::P;
    kf_type["z"] = &signal::KalmanFilter::z;
    kf_type["R"] = &signal::KalmanFilter::R;
    kf_type["Q"] = &signal::KalmanFilter::Q;
    kf_type["F"] = &signal::KalmanFilter::F;
    kf_type["H"] = &signal::KalmanFilter::H;
    kf_type["dim_x"] = &signal::KalmanFilter::dim_x;
    kf_type["dim_z"] = &signal::KalmanFilter::dim_z;
    kf_type["points"] = &signal::KalmanFilter::points;
    kf_type["predict"] = &signal::KalmanFilter::predict;
    kf_type["update"] = &signal::KalmanFilter::update;
    // Factory function for constructing KalmanFilter with optional args
    sig["KalmanFilter_new"] = [](int dim_x, int dim_z, sol::optional<int> dim_u,
                                 sol::optional<int> points,
                                 sol::optional<DType> dtype) {
      return signal::KalmanFilter(dim_x, dim_z, dim_u.value_or(0),
                                  points.value_or(1),
                                  dtype.value_or(DType::F64));
    };

    // Top-level aliases
    sig["convolve"] = &convolve;
    sig["unwrap"] = [](const Array &p, sol::optional<int> axis,
                       sol::optional<double> discont,
                       sol::optional<double> period) {
      return unwrap(p, axis.value_or(-1), discont.value_or(M_PI),
                    period.value_or(2 * M_PI));
    };
    sig["sinc"] = &sinc;
  }

  // Top-level aliases (scipy-compatible)
  m["convolve"] = &convolve;
  m["unwrap"] = [](const Array &p, sol::optional<int> axis,
                   sol::optional<double> discont,
                   sol::optional<double> period) {
    return unwrap(p, axis.value_or(-1), discont.value_or(M_PI),
                  period.value_or(2 * M_PI));
  };
  m["sinc"] = &sinc;

  // ====================================================================
  // Plot (ins.plot.*, requires INSIGHT_USE_MATPLOT)
  // ====================================================================
#ifdef INSIGHT_USE_MATPLOT
  {
    // Ignore SIGPIPE to prevent process crash when gnuplot is unavailable.
    // matplotplusplus spawns gnuplot via popen; if gnuplot is not installed
    // the pipe breaks and SIGPIPE would kill the process.
    std::signal(SIGPIPE, SIG_IGN);

    sol::table plt = m.create_named("plot");

    plt["figure"] = [](sol::optional<int> number) {
      try {
        plot::figure(number.value_or(-1));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["subplot"] = [](int r, int c, int i) {
      try {
        plot::subplot(r, c, i);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["hold"] = [](bool on) {
      try {
        plot::hold(on);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["clf"] = []() {
      try {
        plot::clf();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["show"] = []() {
      try {
        plot::show();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["save"] = [](const std::string &fn) {
      try {
        plot::save(fn);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["plot"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::plot(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::plot(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["plot3"] = [](const Array &x, const Array &y, const Array &z,
                      sol::optional<std::string> fmt) {
      try {
        plot::plot3(x, y, z, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["stairs"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::stairs(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::stairs(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["errorbar"] = [](const Array &x, const Array &y, const Array &err,
                         sol::optional<std::string> fmt) {
      try {
        plot::errorbar(x, y, err, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["area"] = sol::overload(
        [](const Array &y) {
          try {
            plot::area(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y) {
          try {
            plot::area(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["fill"] = [](const Array &x, const Array &y,
                     sol::optional<std::string> color) {
      try {
        plot::fill(x, y, color.value_or("b"));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["line"] = [](const Array &x, const Array &y,
                     sol::optional<std::string> fmt) {
      try {
        plot::line(x, y, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["semilogx"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::semilogx(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::semilogx(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["semilogy"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::semilogy(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::semilogy(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["loglog"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::loglog(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::loglog(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["scatter"] = [](const Array &x, const Array &y,
                        sol::optional<double> size) {
      try {
        plot::scatter(x, y, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["scatter3"] = [](const Array &x, const Array &y, const Array &z,
                         sol::optional<double> size) {
      try {
        plot::scatter3(x, y, z, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["binscatter"] = [](const Array &x, const Array &y) {
      try {
        plot::binscatter(x, y);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["bar"] = sol::overload(
        [](const Array &y, sol::optional<double> w) {
          try {
            plot::bar(y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<double> w) {
          try {
            plot::bar(x, y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["barh"] = sol::overload(
        [](const Array &y, sol::optional<double> w) {
          try {
            plot::barh(y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<double> w) {
          try {
            plot::barh(x, y, w.value_or(0.8));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["barstacked"] = sol::overload(
        [](const Array &Y) {
          try {
            plot::barstacked(Y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &Y) {
          try {
            plot::barstacked(x, Y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["pareto"] = sol::overload(
        [](const Array &y) {
          try {
            plot::pareto(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y) {
          try {
            plot::pareto(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["hist"] = [](const Array &data, sol::optional<int> bins) {
      try {
        plot::hist(data, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["hist2"] = [](const Array &x, const Array &y, sol::optional<int> bins) {
      try {
        plot::hist2(x, y, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["boxplot"] = sol::overload(
        [](const Array &data) {
          try {
            plot::boxplot(data);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &data, std::vector<std::string> groups) {
          try {
            plot::boxplot(data, groups);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["heatmap"] = [](const Array &data) {
      try {
        plot::heatmap(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["stem"] = sol::overload(
        [](const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::stem(y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y, sol::optional<std::string> fmt) {
          try {
            plot::stem(x, y, fmt.value_or(""));
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["stem3"] = [](const Array &x, const Array &y, const Array &z,
                      sol::optional<std::string> fmt) {
      try {
        plot::stem3(x, y, z, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["contour"] = [](const Array &X, const Array &Y, const Array &Z,
                        sol::optional<int> levels) {
      try {
        plot::contour(X, Y, Z, levels.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["contourf"] = [](const Array &X, const Array &Y, const Array &Z,
                         sol::optional<int> levels) {
      try {
        plot::contourf(X, Y, Z, levels.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["pcolor"] = [](const Array &C) {
      try {
        plot::pcolor(C);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["surf"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::surf(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["surfc"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::surfc(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["mesh"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::mesh(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["meshc"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::meshc(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["meshz"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::meshz(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["waterfall"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::waterfall(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ribbon"] = sol::overload(
        [](const Array &y) {
          try {
            plot::ribbon(y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, const Array &y) {
          try {
            plot::ribbon(x, y);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });
    plt["fence"] = [](const Array &X, const Array &Y, const Array &Z) {
      try {
        plot::fence(X, Y, Z);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["quiver"] = [](const Array &X, const Array &Y, const Array &U,
                       const Array &V, sol::optional<double> scale) {
      try {
        plot::quiver(X, Y, U, V, scale.value_or(1.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["quiver3"] = [](const Array &X, const Array &Y, const Array &Z,
                        const Array &U, const Array &V, const Array &W) {
      try {
        plot::quiver3(X, Y, Z, U, V, W);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["feather"] = [](const Array &u, const Array &v) {
      try {
        plot::feather(u, v);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["polarplot"] = [](const Array &theta, const Array &rho,
                          sol::optional<std::string> fmt) {
      try {
        plot::polarplot(theta, rho, fmt.value_or(""));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["polarscatter"] = [](const Array &theta, const Array &rho,
                             sol::optional<double> size) {
      try {
        plot::polarscatter(theta, rho, size.value_or(20.0));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["polarhistogram"] = [](const Array &data, sol::optional<int> bins) {
      try {
        plot::polarhistogram(data, bins.value_or(10));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["compass"] = [](const Array &u, const Array &v) {
      try {
        plot::compass(u, v);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ezpolar"] = [](const std::string &expr) {
      try {
        plot::ezpolar(expr);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["pie"] = sol::overload(
        [](const Array &x) {
          try {
            plot::pie(x);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        },
        [](const Array &x, std::vector<std::string> labels) {
          try {
            plot::pie(x, labels);
          } catch (const std::exception &e) {
            throw sol::error(e.what());
          }
        });

    plt["imshow"] = [](const Array &data) {
      try {
        plot::imshow(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["image"] = [](const Array &data) {
      try {
        plot::image(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["imagesc"] = [](const Array &data) {
      try {
        plot::imagesc(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["graph"] = [](const std::vector<int> &s, const std::vector<int> &t) {
      try {
        plot::graph(s, t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["digraph"] = [](const std::vector<int> &s, const std::vector<int> &t) {
      try {
        plot::digraph(s, t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["parallelplot"] = [](const Array &data) {
      try {
        plot::parallelplot(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["plotmatrix"] = [](const Array &X) {
      try {
        plot::plotmatrix(X);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["wordcloud"] = [](const std::vector<std::string> &w,
                          const std::vector<double> &s) {
      try {
        plot::wordcloud(w, s);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["rgbplot"] = [](const Array &data) {
      try {
        plot::rgbplot(data);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["title"] = [](const std::string &text) {
      try {
        plot::title(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["subtitle"] = [](const std::string &text) {
      try {
        plot::subtitle(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xlabel"] = [](const std::string &text) {
      try {
        plot::xlabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ylabel"] = [](const std::string &text) {
      try {
        plot::ylabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zlabel"] = [](const std::string &text) {
      try {
        plot::zlabel(text);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["legend"] = [](const std::vector<std::string> &labels) {
      try {
        plot::legend(labels);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["xlim"] = [](double xmin, double xmax) {
      try {
        plot::xlim(xmin, xmax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ylim"] = [](double ymin, double ymax) {
      try {
        plot::ylim(ymin, ymax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zlim"] = [](double zmin, double zmax) {
      try {
        plot::zlim(zmin, zmax);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_equal"] = []() {
      try {
        plot::axis_equal();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_tight"] = []() {
      try {
        plot::axis_tight();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_off"] = []() {
      try {
        plot::axis_off();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["axis_ij"] = []() {
      try {
        plot::axis_ij();
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["grid"] = [](bool on) {
      try {
        plot::grid(on);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["xticks"] = [](const std::vector<double> &t) {
      try {
        plot::xticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["yticks"] = [](const std::vector<double> &t) {
      try {
        plot::yticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["zticks"] = [](const std::vector<double> &t) {
      try {
        plot::zticks(t);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xticklabels"] = [](const std::vector<std::string> &l) {
      try {
        plot::xticklabels(l);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["yticklabels"] = [](const std::vector<std::string> &l) {
      try {
        plot::yticklabels(l);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["xtickangle"] = [](double angle) {
      try {
        plot::xtickangle(angle);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ytickangle"] = [](double angle) {
      try {
        plot::ytickangle(angle);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["colormap"] = [](plot::Colormap cmap) {
      try {
        plot::colormap(cmap);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["colorbar"] = [](sol::optional<bool> on) {
      try {
        plot::colorbar(on.value_or(true));
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };

    plt["text"] = [](double x, double y, const std::string &s) {
      try {
        plot::text(x, y, s);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["arrow"] = [](double x1, double y1, double x2, double y2) {
      try {
        plot::arrow(x1, y1, x2, y2);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["rectangle"] = [](double x, double y, double w, double h) {
      try {
        plot::rectangle(x, y, w, h);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["ellipse"] = [](double cx, double cy, double rx, double ry) {
      try {
        plot::ellipse(cx, cy, rx, ry);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
    plt["view"] = [](double az, double el) {
      try {
        plot::view(az, el);
      } catch (const std::exception &e) {
        throw sol::error(e.what());
      }
    };
  }
#endif // INSIGHT_USE_MATPLOT

  // ====================================================================
  // Random
  // ====================================================================
  m["rand"] = [](sol::table shape, sol::optional<DType> dtype,
                 sol::optional<Place> place) {
    return rand(Shape(table_to_shape(shape)), dtype.value_or(DType::F32),
                place.value_or(get_device()));
  };
  m["randn"] = [](sol::table shape, sol::optional<DType> dtype,
                  sol::optional<Place> place) {
    return randn(Shape(table_to_shape(shape)), dtype.value_or(DType::F32),
                 place.value_or(get_device()));
  };
  m["randint"] = [](int64_t low, int64_t high, sol::table shape,
                    sol::optional<DType> dtype, sol::optional<Place> place) {
    return randint(low, high, Shape(table_to_shape(shape)),
                   dtype.value_or(DType::I64), place.value_or(get_device()));
  };
  m["randperm"] = [](int64_t n, sol::optional<DType> dtype,
                     sol::optional<Place> place) {
    return randperm(n, dtype.value_or(DType::I64),
                    place.value_or(get_device()));
  };
  m["seed"] = [](uint64_t s) { return seed(s); };
  m["get_seed"] = []() { return get_seed(); };
  m["rand_like"] = [](const Array &x) { return rand_like(x); };
  m["randn_like"] = [](const Array &x) { return randn_like(x); };
  m["exponential"] = [](double scale, sol::table shape,
                        sol::optional<DType> dtype,
                        sol::optional<Place> place) {
    return exponential(scale, Shape(table_to_shape(shape)),
                       dtype.value_or(DType::F32),
                       place.value_or(get_device()));
  };
  m["gamma"] = [](double shape_param, double rate, sol::table shape,
                  sol::optional<DType> dtype, sol::optional<Place> place) {
    return gamma(shape_param, rate, Shape(table_to_shape(shape)),
                 dtype.value_or(DType::F32), place.value_or(get_device()));
  };
  m["beta"] = [](double a, double b, sol::table shape,
                 sol::optional<DType> dtype, sol::optional<Place> place) {
    return beta(a, b, Shape(table_to_shape(shape)), dtype.value_or(DType::F32),
                place.value_or(get_device()));
  };
  m["binomial"] = [](int64_t n, double p, sol::table shape,
                     sol::optional<DType> dtype, sol::optional<Place> place) {
    return binomial(n, p, Shape(table_to_shape(shape)),
                    dtype.value_or(DType::I64), place.value_or(get_device()));
  };
  m["poisson"] = [](double lam, sol::table shape, sol::optional<DType> dtype,
                    sol::optional<Place> place) {
    return poisson(lam, Shape(table_to_shape(shape)),
                   dtype.value_or(DType::I64), place.value_or(get_device()));
  };

  // ====================================================================
  // Cast
  // ====================================================================
  m["cast"] = [](const Array &input, DType target_dtype,
                 sol::optional<bool> copy) {
    return cast(input, target_dtype, copy.value_or(true));
  };

  // ====================================================================
  // Indexing
  // ====================================================================
  sol::usertype<UniqueResult> ur_type = m.new_usertype<UniqueResult>(
      "UniqueResult", sol::constructors<UniqueResult()>());
  ur_type["unique"] = &UniqueResult::unique;
  ur_type["indices"] = &UniqueResult::indices;
  ur_type["inverse"] = &UniqueResult::inverse;
  ur_type["counts"] = &UniqueResult::counts;

  m["take"] = &take;
  m["nonzero"] = &nonzero;
  m["flatnonzero"] = &flatnonzero;
  m["masked_select"] = &masked_select;
  m["argsort"] = [](const Array &x, sol::optional<int> axis,
                    sol::optional<bool> descending) {
    return argsort(x, axis.value_or(-1), descending.value_or(false));
  };
  m["searchsorted"] = [](const Array &x, const Array &v,
                         sol::optional<std::string> side) {
    return searchsorted(x, v, side.value_or("left"));
  };
  m["sort"] = [](const Array &x, sol::optional<int> axis,
                 sol::optional<bool> descending) {
    return sort(x, axis.value_or(-1), descending.value_or(false));
  };
  m["unique"] = [](const Array &x, sol::optional<bool> return_indices,
                   sol::optional<bool> return_inverse,
                   sol::optional<bool> return_counts) {
    return unique(x, return_indices.value_or(false),
                  return_inverse.value_or(false),
                  return_counts.value_or(false));
  };
  m["topk"] = [](const Array &x, int64_t k, sol::optional<int> axis,
                 sol::optional<bool> largest, sol::optional<bool> sorted) {
    return topk(x, k, axis.value_or(-1), largest.value_or(true),
                sorted.value_or(true));
  };
  m["gather"] = [](const Array &x, int dim, const Array &index) {
    return gather(x, dim, index);
  };
  m["scatter"] = [](const Array &x, int dim, const Array &index,
                    const Array &src) { return scatter(x, dim, index, src); };
  m["scatter_add"] = [](const Array &x, int dim, const Array &index,
                        const Array &src) {
    return scatter_add(x, dim, index, src);
  };
  m["scatter_reduce"] = [](const Array &x, int dim, const Array &index,
                           const Array &src,
                           sol::optional<std::string> reduce) {
    return scatter_reduce(x, dim, index, src, reduce.value_or("replace"));
  };
  m["interp"] = [](const Array &x, const Array &xp, const Array &fp,
                   sol::optional<double> left, sol::optional<double> right) {
    std::optional<double> l =
        left ? std::optional<double>(*left) : std::nullopt;
    std::optional<double> r =
        right ? std::optional<double>(*right) : std::nullopt;
    return interp(x, xp, fp, l, r);
  };
  m["indices"] = [](const Shape &shape, sol::optional<bool> sparse) {
    return indices(shape, sparse.value_or(false));
  };
  m["ix_"] = [](sol::table arrays) { return ix_(table_to_arrays(arrays)); };

  // Push module table onto stack
  sol::stack::push(lua, m);
  return 1;
}
