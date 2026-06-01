--- Radar signal processing functions.
-- Provides pulse compression, Doppler processing, CFAR detection,
-- MVDR beamforming, and ambiguity function computation.
-- @module insight.signal.radar

local native = require("_insight")
local sig = native.signal

local M = {}

local function _wrap(names, fn)
  return function(...)
    if select("#", ...) == 1 and type(select(1, ...)) == "table" then
      local t = select(1, ...)
      local has_names = false
      for k, _ in pairs(t) do
        if type(k) ~= "number" then
          has_names = true
          break
        end
      end
      if has_names then
        local pos = {}
        for i, name in ipairs(names) do
          pos[i] = t[name]
          if pos[i] == nil then
            pos[i] = t[i]
          end
        end
        return fn(table.unpack(pos, 1, #names))
      end
    end
    return fn(...)
  end
end

--- Perform pulse compression using matched filtering.
-- @tparam Array signal Input radar signal (1-D).
-- @tparam Array reference Reference waveform (1-D).
-- @treturn Array Compressed pulse.
M.pulse_compression = _wrap({ "signal", "reference" }, function(signal, reference)
  return sig.pulse_compression(signal, reference)
end)

--- Perform pulse-Doppler processing.
-- @tparam Array data Input CPI data (2-D: pulses x range bins).
-- @int nfft Number of FFT points for Doppler processing.
-- @treturn Array Doppler-range map.
M.pulse_doppler = _wrap({ "data", "nfft" }, function(data, nfft)
  return sig.pulse_doppler(data, nfft)
end)

--- Compute CFAR threshold alpha coefficient.
-- @int n Number of reference cells.
-- @number pfa Probability of false alarm.
-- @number CFAR alpha threshold coefficient.
M.cfar_alpha = _wrap({ "n", "pfa" }, function(n, pfa)
  return sig.cfar_alpha(n, pfa)
end)

--- Apply Cell-Averaging CFAR detection.
-- @tparam Array data Input signal array.
-- @int num_train Number of training (reference) cells.
-- @int num_guard Number of guard cells.
-- @number pfa Probability of false alarm.
-- @treturn Array Binary detection map.
M.ca_cfar = _wrap({ "data", "num_train", "num_guard", "pfa" }, function(data, num_train, num_guard, pfa)
  return sig.ca_cfar(data, num_train, num_guard, pfa)
end)

--- Perform MVDR (Capon) beamforming.
-- @tparam Array x Input data matrix (sensors x samples).
-- @tparam Array steering Steering vector.
-- @treturn Array MVDR beamformer output.
M.mvdr = _wrap({ "x", "steering" }, function(x, steering)
  return sig.mvdr(x, steering)
end)

--- Compute the ambiguity function.
-- @tparam Array x Input signal (1-D).
-- @int max_delay Maximum delay in samples.
-- @int max_doppler Maximum Doppler in samples.
-- @treturn Array Ambiguity function (2-D: delay x Doppler).
M.ambgfun = _wrap({ "x", "max_delay", "max_doppler" }, function(x, max_delay, max_doppler)
  return sig.ambgfun(x, max_delay, max_doppler)
end)

return M
