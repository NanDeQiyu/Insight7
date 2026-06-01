--- Acoustic scale functions.
-- Provides conversions between Hz, Mel, and Bark frequency scales.
-- @module insight.signal.acoustics

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Convert Mel scale to Hz.
-- @number mel Frequency in Mel scale.
-- @treturn number Frequency in Hz.
M.mel2hz = _wrap({ "mel" }, function(mel)
  return sig.mel2hz(mel)
end)

--- Convert Hz to Mel scale.
-- @number hz Frequency in Hz.
-- @treturn number Frequency in Mel scale.
M.hz2mel = _wrap({ "hz" }, function(hz)
  return sig.hz2mel(hz)
end)

--- Compute Mel-spaced frequencies.
-- @int num Number of frequencies to generate.
-- @number[opt=0] fmin Minimum frequency in Hz.
-- @number[opt] fmax Maximum frequency in Hz (default: fs/2).
-- @bool[opt=false] htk Use HTK formula.
-- @treturn Array Mel-spaced frequencies.
M.mel_frequencies = _wrap({ "num", "fmin", "fmax", "htk" }, function(num, fmin, fmax, htk)
  return sig.mel_frequencies(num, fmin or 0, fmax or 8000, htk or false)
end)

--- Convert Hz to Bark scale.
-- @number hz Frequency in Hz.
-- @treturn number Frequency in Bark scale.
M.hz2bark = _wrap({ "hz" }, function(hz)
  return sig.hz2bark(hz)
end)

--- Convert Bark scale to Hz.
-- @number bark Frequency in Bark scale.
-- @treturn number Frequency in Hz.
M.bark2hz = _wrap({ "bark" }, function(bark)
  return sig.bark2hz(bark)
end)

return M
