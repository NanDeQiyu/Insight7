--- B-spline basis functions.
-- Provides Gaussian, cubic, and quadratic B-spline functions.
-- @module insight.signal.bsplines

local native = require("_insight")
local sig = native.signal

local M = {}

--- Return a Gaussian B-spline basis function.
-- @int n Order of the spline.
-- @number x Evaluation points.
-- @treturn Array B-spline values.
function M.gauss_spline(n, x)
  return sig.gauss_spline(n, x)
end

--- Return a cubic B-spline.
-- @number x Evaluation points.
-- @treturn Array Cubic B-spline values.
function M.cubic(x)
  return sig.cubic(x)
end

--- Return a quadratic B-spline.
-- @number x Evaluation points.
-- @treturn Array Quadratic B-spline values.
function M.quadratic(x)
  return sig.quadratic(x)
end

return M
