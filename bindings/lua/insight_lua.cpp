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

#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/slice.h"
#include "insight/init.h"
#include "insight/ops/cast.h"
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

// Helper: Array __tostring
static std::string array_tostring(const Array &a) {
  if (!a.defined())
    return "<insight.Array (undefined)>";
  std::string s = "<insight.Array shape=(";
  for (int i = 0; i < a.shape().ndim(); i++) {
    if (i > 0)
      s += ", ";
    s += std::to_string(a.shape().dim(i));
  }
  s += ") dtype=";
  switch (a.dtype()) {
  case DType::F32:
    s += "float32";
    break;
  case DType::F64:
    s += "float64";
    break;
  case DType::I32:
    s += "int32";
    break;
  case DType::I64:
    s += "int64";
    break;
  case DType::BOOL:
    s += "bool";
    break;
  default:
    s += "other";
    break;
  }
  s += " place=";
  s += (a.place().is_gpu() ? "gpu" : "cpu");
  s += ">";
  return s;
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
  array_type["shape"] = sol::property([](const Array &a) {
    sol::table t;
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
  m["isnan"] = [](const Array &x) { return isnan(x); };
  m["isinf"] = [](const Array &x) { return isinf(x); };
  m["isfinite"] = [](const Array &x) { return isfinite(x); };
  m["where"] = [](const Array &cond, const Array &x, const Array &y) {
    return where(cond, x, y);
  };

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
  m["cumsum"] = [](const Array &x, int axis) { return cumsum(x, axis); };
  m["cumprod"] = [](const Array &x, int axis) { return cumprod(x, axis); };

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

  // ====================================================================
  // Signal
  // ====================================================================
  m["convolve"] = &convolve;
  m["unwrap"] = &unwrap;
  m["sinc"] = &sinc;

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
  m["randperm"] = &randperm;

  // ====================================================================
  // Cast
  // ====================================================================
  m["cast"] = &cast;

  // ====================================================================
  // Indexing
  // ====================================================================
  m["take"] = &take;
  m["nonzero"] = &nonzero;
  m["argsort"] = &argsort;
  m["sort"] = &sort;

  // Push module table onto stack
  sol::stack::push(lua, m);
  return 1;
}
