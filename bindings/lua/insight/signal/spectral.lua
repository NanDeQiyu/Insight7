--- Spectral analysis functions.
-- Provides Welch's method, periodogram, cross-spectral density,
-- coherence, spectrogram, STFT, and vector strength.
-- @module insight.signal.spectral

local native = require("_insight")
local sig = native.signal

local M = {}

--- Estimate power spectral density using Welch's method.
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @string[opt="density"] scaling 'density' or 'spectrum'.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxx' (PSD).
function M.welch(x, fs, window, nperseg, noverlap, nfft, scaling)
  return sig.welch(x, fs or 1.0, window or "hann", nperseg or 256,
    noverlap or 128, nfft or 256, scaling or "density")
end

--- Estimate power spectral density using a periodogram.
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nfft FFT length.
-- @string[opt="density"] scaling 'density' or 'spectrum'.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxx' (PSD).
function M.periodogram(x, fs, window, nfft, scaling)
  return sig.periodogram(x, fs or 1.0, window or "hann", nfft or 256,
    scaling or "density")
end

--- Estimate cross-spectral density using Welch's method.
-- @tparam Array x First input signal.
-- @tparam Array y Second input signal.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies) and 'Pxy' (CSD).
function M.csd(x, y, fs, window, nperseg, noverlap, nfft)
  return sig.csd(x, y, fs or 1.0, window or "hann", nperseg or 256,
    noverlap or 128, nfft or 256)
end

--- Estimate magnitude-squared coherence using Welch's method.
-- @tparam Array x First input signal.
-- @tparam Array y Second input signal.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies) and 'Cxy' (coherence).
function M.coherence(x, y, fs, window, nperseg, noverlap, nfft)
  return sig.coherence(x, y, fs or 1.0, window or "hann", nperseg or 256,
    noverlap or 128, nfft or 256)
end

--- Compute a spectrogram (SSTFT power spectral density).
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies), 't' (times), 'Sxx' (spectrogram).
function M.spectrogram(x, fs, window, nperseg, noverlap, nfft)
  return sig.spectrogram(x, fs or 1.0, window or "hann", nperseg or 256,
    noverlap or 128, nfft or 256)
end

--- Compute the Short-Time Fourier Transform (STFT).
-- @tparam Array x Input signal array.
-- @number[opt=1.0] fs Sampling frequency.
-- @string[opt="hann"] window Window type.
-- @int[opt] nperseg Length of each segment.
-- @int[opt] noverlap Number of overlapping points.
-- @int[opt] nfft FFT length.
-- @treturn table Table with fields 'f' (frequencies), 't' (times), 'Zxx' (STFT).
function M.stft(x, fs, window, nperseg, noverlap, nfft)
  return sig.stft(x, fs or 1.0, window or "hann", nperseg or 256,
    noverlap or 128, nfft or 256)
end

--- Compute the vector strength (phase coherence).
-- @tparam Array signal Input signal array.
-- @number freq Frequency in Hz.
-- @number fs Sampling frequency in Hz.
-- @treturn table Table with fields 'strength' and 'angle'.
function M.vectorstrength(signal, freq, fs)
  return sig.vectorstrength(signal, freq, fs)
end

return M
