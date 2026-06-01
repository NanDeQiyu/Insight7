--- Filter design functions.
-- Provides FIR filter design, Kaiser window parameter estimation,
-- and complex sorting utilities.
-- @module insight.signal.filter_design

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Compute the Kaiser beta parameter for a given attenuation.
-- @number attenuation Stopband attenuation in dB.
-- @number Kaiser beta parameter.
M.kaiser_beta = _wrap({ "attenuation" }, function(attenuation)
  return sig.kaiser_beta(attenuation)
end)

--- Compute the Kaiser window attenuation for given parameters.
-- @int n Number of taps.
-- @number width Transition width (normalized).
-- @number Attenuation in dB.
M.kaiser_atten = _wrap({ "n", "width" }, function(n, width)
  return sig.kaiser_atten(n, width)
end)

--- Design a FIR filter using the window method.
-- @int numtaps Number of filter taps.
-- @number cutoff Cutoff frequency (normalized, 0 to 1).
-- @string[opt="lowpass"] type Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'.
-- @bool[opt=false] pass_zero If true, the DC gain is nonzero at frequency 0.
-- @treturn Array FIR filter coefficients.
M.firwin = _wrap({ "numtaps", "cutoff", "type", "pass_zero" }, function(numtaps, cutoff, type, pass_zero)
  return sig.firwin(numtaps, cutoff, type or "lowpass", pass_zero or false)
end)

--- Design a FIR filter using frequency sampling.
-- @int numtaps Number of filter taps.
-- @tparam table freq Frequency breakpoints (normalized, 0 to 1).
-- @tparam table gain Desired gain at each frequency breakpoint.
-- @treturn Array FIR filter coefficients.
M.firwin2 = _wrap({ "numtaps", "freq", "gain" }, function(numtaps, freq, gain)
  return sig.firwin2(numtaps, freq, gain)
end)

--- Sort complex numbers by real part, then imaginary part.
-- @tparam Array x Complex input array.
-- @treturn Array Sorted complex array.
M.cmplx_sort = _wrap({ "x" }, function(x)
  return sig.cmplx_sort(x)
end)

return M
