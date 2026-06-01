--- Elementwise binary operations.
--
-- Provides arithmetic, comparison, and logical operations
-- that are applied element-wise to arrays.
--
-- @module insight.elementwise

local native = require("_insight")
local M = {}

--- Element-wise addition.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise sum.
function M.add(a, b)
  return native.add(a, b)
end

--- Element-wise subtraction.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise difference.
function M.sub(a, b)
  return native.sub(a, b)
end

--- Element-wise multiplication.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise product.
function M.mul(a, b)
  return native.mul(a, b)
end

--- Element-wise division.
-- @tparam Array a First input array (numerator).
-- @tparam Array b Second input array (denominator).
-- @treturn Array Element-wise quotient.
function M.div(a, b)
  return native.div(a, b)
end

--- Element-wise exponentiation.
-- @tparam Array a Base array.
-- @tparam Array b Exponent array.
-- @treturn Array Element-wise power.
function M.pow(a, b)
  return native.pow(a, b)
end

--- Element-wise modulus.
-- @tparam Array a First input array.
-- @tparam Array b Second input array (divisor).
-- @treturn Array Element-wise remainder.
function M.mod(a, b)
  return native.mod(a, b)
end

--- Element-wise maximum of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise maximum.
function M.maximum(a, b)
  return native.maximum(a, b)
end

--- Element-wise minimum of two arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Element-wise minimum.
function M.minimum(a, b)
  return native.minimum(a, b)
end

--- Element-wise equality comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array of element-wise equality.
function M.equal(a, b)
  return native.equal(a, b)
end

--- Element-wise inequality comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array of element-wise inequality.
function M.not_equal(a, b)
  return native.not_equal(a, b)
end

--- Element-wise greater-than comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a > b.
function M.greater(a, b)
  return native.greater(a, b)
end

--- Element-wise less-than comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a < b.
function M.less(a, b)
  return native.less(a, b)
end

--- Element-wise greater-or-equal comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a >= b.
function M.greater_equal(a, b)
  return native.greater_equal(a, b)
end

--- Element-wise less-or-equal comparison.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @treturn Array Boolean array where a <= b.
function M.less_equal(a, b)
  return native.less_equal(a, b)
end

--- Element-wise logical AND.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise AND.
function M.logical_and(a, b)
  return native.logical_and(a, b)
end

--- Element-wise logical OR.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise OR.
function M.logical_or(a, b)
  return native.logical_or(a, b)
end

--- Element-wise logical XOR.
-- @tparam Array a First boolean input array.
-- @tparam Array b Second boolean input array.
-- @treturn Array Boolean array of element-wise XOR.
function M.logical_xor(a, b)
  return native.logical_xor(a, b)
end

--- Element-wise logical NOT.
-- @tparam Array x Input boolean array.
-- @treturn Array Boolean array of element-wise NOT.
function M.logical_not(x)
  return native.logical_not(x)
end

return M
