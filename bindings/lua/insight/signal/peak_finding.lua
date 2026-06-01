--- Peak finding functions.
-- Provides functions to find local extrema (maxima and minima)
-- in signal arrays.
-- @module insight.signal.peak_finding

local native = require("_insight")
local sig = native.signal

local M = {}

--- Find indices of local extrema.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local extrema.
function M.argrelextrema(x, order)
  return sig.argrelextrema(x, order)
end

--- Find indices of local maxima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local maxima.
function M.argrelmax(x, order)
  return sig.argrelmax(x, order)
end

--- Find indices of local minima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local minima.
function M.argrelmin(x, order)
  return sig.argrelmin(x, order)
end

return M
