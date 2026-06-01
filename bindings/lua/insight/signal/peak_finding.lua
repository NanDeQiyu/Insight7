--- Peak finding functions.
-- Provides functions to find local extrema (maxima and minima)
-- in signal arrays.
-- @module insight.signal.peak_finding

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Find indices of local extrema.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local extrema.
M.argrelextrema = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelextrema(x, order)
end)

--- Find indices of local maxima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local maxima.
M.argrelmax = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelmax(x, order)
end)

--- Find indices of local minima.
-- @tparam Array x Input signal array.
-- @int order Number of points on each side to compare.
-- @treturn Array Indices of local minima.
M.argrelmin = _wrap({ "x", "order" }, function(x, order)
  return sig.argrelmin(x, order)
end)

return M
