--- Acoustic scale functions.
-- Provides conversions between Hz, Mel, and Bark frequency scales.
-- @module insight.signal.acoustics

local native = require("_insight")
local sig = native.signal

local M = {}

--- Convert Mel scale to Hz.
-- @number mel Frequency in Mel scale.
-- @treturn number Frequency in Hz.
function M.mel2hz(mel)
  return sig.mel2hz(mel)
end

--- Convert Hz to Mel scale.
-- @number hz Frequency in Hz.
-- @treturn number Frequency in Mel scale.
function M.hz2mel(hz)
  return sig.hz2mel(hz)
end

--- Compute Mel-spaced frequencies.
-- @int num Number of frequencies to generate.
-- @number[opt=0] fmin Minimum frequency in Hz.
-- @number[opt] fmax Maximum frequency in Hz (default: fs/2).
-- @bool[opt=false] htk Use HTK formula.
-- @treturn Array Mel-spaced frequencies.
function M.mel_frequencies(num, fmin, fmax, htk)
  return sig.mel_frequencies(num, fmin or 0, fmax or 8000, htk or false)
end

--- Convert Hz to Bark scale.
-- @number hz Frequency in Hz.
-- @treturn number Frequency in Bark scale.
function M.hz2bark(hz)
  return sig.hz2bark(hz)
end

--- Convert Bark scale to Hz.
-- @number bark Frequency in Bark scale.
-- @treturn number Frequency in Hz.
function M.bark2hz(bark)
  return sig.bark2hz(bark)
end

return M
