--- Radar signal processing functions.
-- Provides pulse compression, Doppler processing, CFAR detection,
-- MVDR beamforming, and ambiguity function computation.
-- @module insight.signal.radar

local native = require("_insight")
local sig = native.signal

local M = {}

--- Perform pulse compression using matched filtering.
-- @tparam Array signal Input radar signal (1-D).
-- @tparam Array reference Reference waveform (1-D).
-- @treturn Array Compressed pulse.
function M.pulse_compression(signal, reference)
  return sig.pulse_compression(signal, reference)
end

--- Perform pulse-Doppler processing.
-- @tparam Array data Input CPI data (2-D: pulses x range bins).
-- @int nfft Number of FFT points for Doppler processing.
-- @treturn Array Doppler-range map.
function M.pulse_doppler(data, nfft)
  return sig.pulse_doppler(data, nfft)
end

--- Compute CFAR threshold alpha coefficient.
-- @int n Number of reference cells.
-- @number pfa Probability of false alarm.
-- @number CFAR alpha threshold coefficient.
function M.cfar_alpha(n, pfa)
  return sig.cfar_alpha(n, pfa)
end

--- Apply Cell-Averaging CFAR detection.
-- @tparam Array data Input signal array.
-- @int num_train Number of training (reference) cells.
-- @int num_guard Number of guard cells.
-- @number pfa Probability of false alarm.
-- @treturn Array Binary detection map.
function M.ca_cfar(data, num_train, num_guard, pfa)
  return sig.ca_cfar(data, num_train, num_guard, pfa)
end

--- Perform MVDR (Capon) beamforming.
-- @tparam Array x Input data matrix (sensors x samples).
-- @tparam Array steering Steering vector.
-- @treturn Array MVDR beamformer output.
function M.mvdr(x, steering)
  return sig.mvdr(x, steering)
end

--- Compute the ambiguity function.
-- @tparam Array x Input signal (1-D).
-- @int max_delay Maximum delay in samples.
-- @int max_doppler Maximum Doppler in samples.
-- @treturn Array Ambiguity function (2-D: delay x Doppler).
function M.ambgfun(x, max_delay, max_doppler)
  return sig.ambgfun(x, max_delay, max_doppler)
end

return M
