--- Wavelet functions.
-- Provides Morlet, Ricker (Mexican hat), and Morlet2 wavelet functions.
-- @module insight.signal.wavelets

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Compute a Morlet wavelet.
-- @int M Length of the wavelet (number of points).
-- @number[opt=5.0] w Frequency parameter (omega0).
-- @number[opt=1.0] s Scaling factor.
-- @bool[opt=true] complete Whether to use the complete version.
-- @treturn Array Complex Morlet wavelet.
M.morlet = _wrap({ "M", "w", "s", "complete" }, function(M_val, w, s, complete)
  return sig.morlet(M_val, w, s, complete)
end)

--- Compute a Ricker (Mexican hat) wavelet.
-- @int points Number of points.
-- @number a Width parameter.
-- @treturn Array Ricker wavelet.
M.ricker = _wrap({ "points", "a" }, function(points, a)
  return sig.ricker(points, a)
end)

--- Compute a Morlet2 wavelet.
-- @int M Length of the wavelet (number of points).
-- @number s Scale parameter.
-- @number[opt=5.0] w Frequency parameter (omega0).
-- @treturn Array Complex Morlet2 wavelet.
M.morlet2 = _wrap({ "M", "s", "w" }, function(M_val, s, w)
  return sig.morlet2(M_val, s, w)
end)

return M
