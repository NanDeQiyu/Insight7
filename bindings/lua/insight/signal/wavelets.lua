--- Wavelet functions.
-- Provides Morlet, Ricker (Mexican hat), and Morlet2 wavelet functions.
-- @module insight.signal.wavelets

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Compute a Morlet wavelet.
-- @number w Frequency parameter.
-- @treturn Array Complex Morlet wavelet.
M.morlet = _wrap({ "w" }, function(w)
  return sig.morlet(w)
end)

--- Compute a Ricker (Mexican hat) wavelet.
-- @int points Number of points.
-- @number a Width parameter.
-- @treturn Array Ricker wavelet.
M.ricker = _wrap({ "points", "a" }, function(points, a)
  return sig.ricker(points, a)
end)

--- Compute a Morlet2 wavelet.
-- @number w Frequency parameter.
-- @number s Scale parameter.
-- @treturn Array Complex Morlet2 wavelet.
M.morlet2 = _wrap({ "w", "s" }, function(w, s)
  return sig.morlet2(w, s)
end)

return M
