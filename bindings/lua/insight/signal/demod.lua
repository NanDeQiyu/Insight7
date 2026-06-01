--- FM demodulation functions.
-- @module insight.signal.demod

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Demodulate an FM signal.
-- @tparam Array x Input FM signal array.
-- @number fs Sampling frequency in Hz.
-- @treturn Array Demodulated signal.
M.fm_demod = _wrap({ "x", "fs" }, function(x, fs)
  return sig.fm_demod(x, fs)
end)

return M
