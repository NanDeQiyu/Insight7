--- FM demodulation functions.
-- @module insight.signal.demod

local native = require("_insight")
local sig = native.signal

local M = {}

--- Demodulate an FM signal.
-- @tparam Array x Input FM signal array.
-- @number fs Sampling frequency in Hz.
-- @treturn Array Demodulated signal.
function M.fm_demod(x, fs)
  return sig.fm_demod(x, fs)
end

return M
