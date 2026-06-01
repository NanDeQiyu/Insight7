--- Wavelet functions.
-- Provides Morlet, Ricker (Mexican hat), and Morlet2 wavelet functions.
-- @module insight.signal.wavelets

local native = require("_insight")
local sig = native.signal

local M = {}

--- Compute a Morlet wavelet.
-- @number w Frequency parameter.
-- @treturn Array Complex Morlet wavelet.
function M.morlet(w)
  return sig.morlet(w)
end

--- Compute a Ricker (Mexican hat) wavelet.
-- @int points Number of points.
-- @number a Width parameter.
-- @treturn Array Ricker wavelet.
function M.ricker(points, a)
  return sig.ricker(points, a)
end

--- Compute a Morlet2 wavelet.
-- @number w Frequency parameter.
-- @number s Scale parameter.
-- @treturn Array Complex Morlet2 wavelet.
function M.morlet2(w, s)
  return sig.morlet2(w, s)
end

return M
