--- Waveform generation functions.
-- Provides standard signal waveforms such as sawtooth, square,
-- Gaussian pulse, chirp, and unit impulse.
-- @module insight.signal.waveforms

local native = require("_insight")
local sig = native.signal

local M = {}

local _wrap = require("insight._wrap")

--- Return a periodic sawtooth or triangle waveform.
-- @number t Time array or scalar.
-- @number[opt=1.0] width Width of the rising ramp (0 to 1).
-- @treturn Array Sawtooth waveform.
M.sawtooth = _wrap({ "t", "width" }, function(t, width)
  return sig.sawtooth(t, width or 1.0)
end)

--- Return a periodic square-wave waveform.
-- @number t Time array or scalar.
-- @number[opt=0.5] duty Duty cycle (0 to 1).
-- @treturn Array Square waveform.
M.square = _wrap({ "t", "duty" }, function(t, duty)
  return sig.square_wf(t, duty or 0.5)
end)

--- Return a Gaussian pulse.
-- @number t Time array or scalar.
-- @number[opt=1000] fc Center frequency in Hz.
-- @number[opt=1.0] bw Fractional bandwidth.
-- @number[opt=6] bwr Bandwidth reference level in dB.
-- @number[opt=-60] tpr Pulse reference level in dB.
-- @treturn Array Gaussian pulse.
M.gausspulse = _wrap({ "t", "fc", "bw", "bwr", "tpr" }, function(t, fc, bw, bwr, tpr)
  return sig.gausspulse(t, fc, bw, bwr, tpr)
end)

--- Return a chirp signal.
-- @number t Time array or scalar.
-- @number f0 Instantaneous frequency at time 0.
-- @number t1 Time at which f1 is specified.
-- @number f1 Frequency at time t1.
-- @string[opt="linear"] method Chirp method: 'linear', 'quadratic', 'logarithmic', 'hyperbolic'.
-- @number[opt=0] phi Initial phase in degrees.
-- @treturn Array Chirp signal.
M.chirp = _wrap({ "t", "f0", "t1", "f1", "method", "phi" }, function(t, f0, t1, f1, method, phi)
  return sig.chirp(t, f0, t1, f1, method or "linear", phi or 0)
end)

--- Return a unit impulse (delta function).
-- @tparam table|Array shape Shape of the output array or an existing array.
-- @int[opt] idx Index of the impulse (default: center).
-- @treturn Array Unit impulse.
M.unit_impulse = _wrap({ "shape", "idx" }, function(shape, idx)
  if idx then
    return sig.unit_impulse(shape, idx)
  end
  return sig.unit_impulse(shape)
end)

return M
