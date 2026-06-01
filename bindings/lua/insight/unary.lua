--- Unary math operations.
--
-- Provides element-wise unary math functions including
-- trigonometric, exponential, rounding, and classification operations.
--
-- @module insight.unary

local native = require("_insight")
local M = {}

--- Element-wise absolute value.
-- @tparam Array x Input array.
-- @treturn Array Absolute values.
function M.abs(x)
  return native.abs(x)
end

--- Element-wise negation.
-- @tparam Array x Input array.
-- @treturn Array Negated values.
function M.negative(x)
  return native.negative(x)
end

--- Element-wise square.
-- @tparam Array x Input array.
-- @treturn Array Squared values.
function M.square(x)
  return native.square(x)
end

--- Element-wise square root.
-- @tparam Array x Input array.
-- @treturn Array Square root values.
function M.sqrt(x)
  return native.sqrt(x)
end

--- Element-wise exponential (e^x).
-- @tparam Array x Input array.
-- @treturn Array Exponential values.
function M.exp(x)
  return native.exp(x)
end

--- Element-wise natural logarithm.
-- @tparam Array x Input array.
-- @treturn Array Natural log values.
function M.log(x)
  return native.log(x)
end

--- Element-wise base-2 logarithm.
-- @tparam Array x Input array.
-- @treturn Array Base-2 log values.
function M.log2(x)
  return native.log2(x)
end

--- Element-wise base-10 logarithm.
-- @tparam Array x Input array.
-- @treturn Array Base-10 log values.
function M.log10(x)
  return native.log10(x)
end

--- Element-wise sine.
-- @tparam Array x Input array (radians).
-- @treturn Array Sine values.
function M.sin(x)
  return native.sin(x)
end

--- Element-wise cosine.
-- @tparam Array x Input array (radians).
-- @treturn Array Cosine values.
function M.cos(x)
  return native.cos(x)
end

--- Element-wise tangent.
-- @tparam Array x Input array (radians).
-- @treturn Array Tangent values.
function M.tan(x)
  return native.tan(x)
end

--- Element-wise inverse sine (arcsine).
-- @tparam Array x Input array.
-- @treturn Array Arcsine values in radians.
function M.asin(x)
  return native.asin(x)
end

--- Element-wise inverse cosine (arccosine).
-- @tparam Array x Input array.
-- @treturn Array Arccosine values in radians.
function M.acos(x)
  return native.acos(x)
end

--- Element-wise inverse tangent (arctangent).
-- @tparam Array x Input array.
-- @treturn Array Arctangent values in radians.
function M.atan(x)
  return native.atan(x)
end

--- Element-wise hyperbolic sine.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic sine values.
function M.sinh(x)
  return native.sinh(x)
end

--- Element-wise hyperbolic cosine.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic cosine values.
function M.cosh(x)
  return native.cosh(x)
end

--- Element-wise hyperbolic tangent.
-- @tparam Array x Input array.
-- @treturn Array Hyperbolic tangent values.
function M.tanh(x)
  return native.tanh(x)
end

--- Element-wise floor (round down to nearest integer).
-- @tparam Array x Input array.
-- @treturn Array Floored values.
function M.floor(x)
  return native.floor(x)
end

--- Element-wise ceil (round up to nearest integer).
-- @tparam Array x Input array.
-- @treturn Array Ceiled values.
function M.ceil(x)
  return native.ceil(x)
end

--- Element-wise round to nearest integer.
-- @tparam Array x Input array.
-- @treturn Array Rounded values.
function M.round(x)
  return native.round(x)
end

--- Element-wise sign function (-1, 0, or 1).
-- @tparam Array x Input array.
-- @treturn Array Sign values.
function M.sign(x)
  return native.sign(x)
end

--- Element-wise NaN check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is NaN.
function M.isnan(x)
  return native.isnan(x)
end

--- Element-wise infinity check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is infinite.
function M.isinf(x)
  return native.isinf(x)
end

--- Element-wise finite check.
-- @tparam Array x Input array.
-- @treturn Array Boolean array, true where element is finite.
function M.isfinite(x)
  return native.isfinite(x)
end

--- Element-wise base-2 exponential (2^x).
-- @tparam Array x Input array.
-- @treturn Array Base-2 exponential values.
function M.exp2(x)
  return native.exp2(x)
end

--- Element-wise exponential minus 1 (e^x - 1).
-- @tparam Array x Input array.
-- @treturn Array expm1 values.
function M.expm1(x)
  return native.expm1(x)
end

--- Element-wise natural logarithm of 1 + x.
-- @tparam Array x Input array.
-- @treturn Array log1p values.
function M.log1p(x)
  return native.log1p(x)
end

--- Element-wise cube root.
-- @tparam Array x Input array.
-- @treturn Array Cube root values.
function M.cbrt(x)
  return native.cbrt(x)
end

--- Element-wise reciprocal (1/x).
-- @tparam Array x Input array.
-- @treturn Array Reciprocal values.
function M.reciprocal(x)
  return native.reciprocal(x)
end

--- Element-wise inverse hyperbolic sine.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic sine values.
function M.asinh(x)
  return native.asinh(x)
end

--- Element-wise inverse hyperbolic cosine.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic cosine values.
function M.acosh(x)
  return native.acosh(x)
end

--- Element-wise inverse hyperbolic tangent.
-- @tparam Array x Input array.
-- @treturn Array Inverse hyperbolic tangent values.
function M.atanh(x)
  return native.atanh(x)
end

--- Element-wise truncation toward zero.
-- @tparam Array x Input array.
-- @treturn Array Truncated values.
function M.trunc(x)
  return native.trunc(x)
end

--- Convert angles from degrees to radians.
-- @tparam Array x Input array in degrees.
-- @treturn Array Angles in radians.
function M.deg2rad(x)
  return native.deg2rad(x)
end

--- Convert angles from radians to degrees.
-- @tparam Array x Input array in radians.
-- @treturn Array Angles in degrees.
function M.rad2deg(x)
  return native.rad2deg(x)
end

--- Element-wise conditional selection.
-- Selects elements from x where cond is true, else from y.
-- @tparam Array cond Boolean condition array.
-- @tparam Array x Values selected where cond is true.
-- @tparam Array y Values selected where cond is false.
-- @treturn Array Selected values.
function M.where(cond, x, y)
  return native.where(cond, x, y)
end

return M
