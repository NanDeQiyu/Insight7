--- FM demodulation functions.
-- @module insight.signal.demod

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Demodulate an FM signal.
-- @tparam Array x Input FM signal array.
-- @int[opt=-1] axis Axis along which to demodulate. -1 is last axis.
-- @treturn Array Demodulated signal.
M.fm_demod = _wrap({ "x", "axis" }, function(x, axis)
  return sig.fm_demod(x, axis or -1)
end)

return M
