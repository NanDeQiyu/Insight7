--- Unary math operations.
--
-- Provides element-wise unary math functions including
-- trigonometric, exponential, rounding, and classification operations.
--
-- @module insight.unary

local native = require("_insight")
local M = {}

local function _wrap(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end

--- Element-wise absolute value.
-- @tparam Array x Input array.
-- @treturn Array Absolute values.
M.abs = _wrap({ "x" }, function(x)
  return native.abs(x)
end)

--- Element-wise negation.
-- @tparam Array x Input array.
-- @treturn Array Negated values.
M.negative = _wrap({ "x" }, function(x)
  return native.negative(x)
end)

--- Element-wise square.
-- @tparam Array x Input array.
-- @treturn Array Squared values.
M.square = _wrap({ "x" }, function(x)
  return native.square(x)
end)

--- Element-wise square root.
-- @tparam Array x Input array.
-- @treturn Array Square root values.
M.sqrt = _wrap({ "x" }, function(x)
  return native.sqrt(x)
end)

--- Element-wise exponential (e^x).
-- @tparam Array x Input array.
-- @treturn Array Exponential values.
M.exp = _wrap({ "x" }, function(x)
  return native.exp(x)
end)

--- Element-wise natural logarithm.
-- @tparam Array x Input array.
-- @treturn Array Natural log values.
M.log = _wrap({ "x" }, function(x)
  return native.log(x)
end)

--- Element-wise base-2 logarithm.
-- @tparam Array x Input array.
-- @treturn Array Base-2 log values.
M.log2 = _wrap({ "x" }, function(x)
  return native.log2(x)
end)

--- Element-wise base-10 logarithm.
-- @tparam Array x Input array.
-- @treturn Array Base-10 log values.
M.log10 = _wrap({ "x" }, function(x)
  return native.log10(x)
end)

--- Element-wise sine.
-- @tparam Array x Input array (radians).
-- @treturn Array Sine values.
M.sin = _wrap({ "x" }, function(x)
  return native.sin(x)
end)

--- Element-wise cosine.
-- @tparam Array x Input array (radians).
-- @treturn Array Cosine values.
M.cos = _wrap({ "x" }, function(x)
  return native.cos(x)
end)

--- Element-wise tangent.
-- @tparam Array x Input array (radians).
-- @treturn Array Tangent values.
M.tan = _wrap({ "x" }, function(x)
  return native.tan(x)
end)

--- Element-wise inverse sine (arcsine).
-- @tparam Array x Input array.
-- @treturn Array Arcsine values in radians.
M.asin = _wrap({ "x" }, function(x)
  return native.asin(x)
end)

--- Element-wise inverse cosine (arccosine).
-- @tparam Array x Input array.
-- @treturn Array Arccosine values in radians.
M.acos = _wrap({ "x" }, function(x)
  return native.acos(x)
end)

--- Element-wise inverse tangent (arctangent).
-- @tparam Array x Input array.
-- @treturn Array Arctangent values in radians.
M.atan = _wrap({ "x" }, function(x)
  return native.atan(x)
end)

--- Element-wise hyperbolic sine.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic sine values.
M.sinh = _wrap({ "x" }, function(x)
  return native.sinh(x)
end)

--- Element-wise hyperbolic cosine.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic cosine values.
M.cosh = _wrap({ "x" }, function(x)
  return native.cosh(x)
end)

--- Element-wise hyperbolic tangent.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic tangent values.
M.tanh = _wrap({ "x" }, function(x)
  return native.tanh(x)
end)

--- Element-wise floor (round down to nearest integer).
-- @tparam Array x Input array.
-- @treturn Array Floored values.
M.floor = _wrap({ "x" }, function(x)
  return native.floor(x)
end)

--- Element-wise ceil (round up to nearest integer).
-- @tparam Array x Input array.
-- @treturn Array Ceiled values.
M.ceil = _wrap({ "x" }, function(x)
  return native.ceil(x)
end)

--- Element-wise round to nearest integer.
-- @tparam Array x Input array.
-- @treturn Array Rounded values.
M.round = _wrap({ "x" }, function(x)
  return native.round(x)
end)

--- Element-wise sign function (-1, 0, or 1).
-- @tparam Array x Input array.
-- @treturn Array Sign values.
M.sign = _wrap({ "x" }, function(x)
  return native.sign(x)
end)

--- Element-wise NaN check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is NaN.
M.isnan = _wrap({ "x" }, function(x)
  return native.isnan(x)
end)

--- Element-wise infinity check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is infinite.
M.isinf = _wrap({ "x" }, function(x)
  return native.isinf(x)
end)

--- Element-wise finite check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is finite.
M.isfinite = _wrap({ "x" }, function(x)
  return native.isfinite(x)
end)

--- Element-wise base-2 exponential (2^x).
-- @tparam Array x Input array.
-- @treturn Array Base-2 exponential values.
M.exp2 = _wrap({ "x" }, function(x)
  return native.exp2(x)
end)

--- Element-wise exponential minus 1 (e^x - 1).
-- @tparam Array x Input array.
-- @treturn Array expm1 values.
M.expm1 = _wrap({ "x" }, function(x)
  return native.expm1(x)
end)

--- Element-wise natural logarithm of 1 + x.
-- @tparam Array x Input array.
-- @treturn Array log1p values.
M.log1p = _wrap({ "x" }, function(x)
  return native.log1p(x)
end)

--- Element-wise cube root.
-- @tparam Array x Input array.
-- @treturn Array Cube root values.
M.cbrt = _wrap({ "x" }, function(x)
  return native.cbrt(x)
end)

--- Element-wise reciprocal (1/x).
-- @tparam Array x Input array.
-- @treturn Array Reciprocal values.
M.reciprocal = _wrap({ "x" }, function(x)
  return native.reciprocal(x)
end)

--- Element-wise inverse hyperbolic sine.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic sine values.
M.asinh = _wrap({ "x" }, function(x)
  return native.asinh(x)
end)

--- Element-wise inverse hyperbolic cosine.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic cosine values.
M.acosh = _wrap({ "x" }, function(x)
  return native.acosh(x)
end)

--- Element-wise inverse hyperbolic tangent.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic tangent values.
M.atanh = _wrap({ "x" }, function(x)
  return native.atanh(x)
end)

--- Element-wise truncation toward zero.
-- @tparam Array x Input array.
-- @treturn Array Truncated values.
M.trunc = _wrap({ "x" }, function(x)
  return native.trunc(x)
end)

--- Convert angles from degrees to radians.
-- @tparam Array x Input array in degrees.
-- @treturn Array Angles in radians.
M.deg2rad = _wrap({ "x" }, function(x)
  return native.deg2rad(x)
end)

--- Convert angles from radians to degrees.
-- @tparam Array x Input array in radians.
-- @treturn Array Angles in degrees.
M.rad2deg = _wrap({ "x" }, function(x)
  return native.rad2deg(x)
end)

--- Element-wise conditional selection.
-- Selects elements from x where cond is true, else from y.
-- @tparam Array cond Boolean condition array.
-- @tparam Array x Values selected where cond is true.
-- @tparam Array y Values selected where cond is false.
-- @treturn Array Selected values.
M.where = _wrap({ "cond", "x", "y" }, function(cond, x, y)
  return native.where(cond, x, y)
end)

return M
