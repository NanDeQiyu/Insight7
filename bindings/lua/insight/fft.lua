--- Fast Fourier Transform operations.
--
-- Provides 1-D, 2-D, and N-D FFT and inverse FFT, real-input FFT,
-- sample frequency generation, and related utilities.
--
-- @module insight.fft

local native = require("_insight")
local M = {}

--- 1-D discrete Fourier transform.
-- @tparam Array x Input array.
-- @int[opt=-1] n FFT length. -1 uses input length.
-- @int[opt=-1] axis Axis along which to compute. -1 is last axis.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array FFT result.
function M.fft(x, n, axis, norm)
  return native.fft(x, n or -1, axis or -1, norm or "backward")
end

--- 1-D inverse discrete Fourier transform.
-- @tparam Array x Input frequency-domain array.
-- @int[opt=-1] n IFFT length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array IFFT result.
function M.ifft(x, n, axis, norm)
  return native.ifft(x, n or -1, axis or -1, norm or "backward")
end

--- 1-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @int[opt=-1] n FFT length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array Complex FFT result (positive frequencies only).
function M.rfft(x, n, axis, norm)
  return native.rfft(x, n or -1, axis or -1, norm or "backward")
end

--- Inverse of rfft (complex to real).
-- @tparam Array x Complex input from rfft.
-- @int[opt=-1] n Output length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array Real-valued IFFT result.
function M.irfft(x, n, axis, norm)
  return native.irfft(x, n or -1, axis or -1, norm or "backward")
end

--- DFT sample frequencies.
-- @int n Window length.
-- @number[opt=1] d Sample spacing.
-- @treturn Array Frequency vector.
function M.fftfreq(n, d)
  return native.fftfreq(n, d or 1.0)
end

--- DFT sample frequencies for rfft.
-- @int n Window length.
-- @number[opt=1] d Sample spacing.
-- @treturn Array Frequency vector (non-negative only).
function M.rfftfreq(n, d)
  return native.rfftfreq(n, d or 1.0)
end

--- Shift zero-frequency component to center of spectrum.
-- @tparam Array x Input frequency-domain array.
-- @treturn Array Shifted array.
function M.fftshift(x)
  return native.fftshift(x)
end

--- Inverse of fftshift.
-- @tparam Array x Input shifted array.
-- @treturn Array Un-shifted array.
function M.ifftshift(x)
  return native.ifftshift(x)
end

--- Find the next fast FFT length.
-- @int target Target length.
-- @bool[opt=true] real If true, consider real FFT efficiency.
-- @int Fast FFT length >= target.
function M.next_fast_len(target, real)
  return native.next_fast_len(target, real == nil and true or real)
end

--- FFT of a Hermitian signal (real output).
-- @tparam Array x Hermitian-symmetric input.
-- @int[opt=-1] n Output length.
-- @treturn Array Real-valued FFT result.
function M.hfft(x, n)
  return native.hfft(x, n or -1)
end

--- Inverse of hfft.
-- @tparam Array x Real input.
-- @int[opt=-1] n Output length.
-- @treturn Array Hermitian-symmetric output.
function M.ihfft(x, n)
  return native.ihfft(x, n or -1)
end

--- 2-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @treturn Array Complex 2-D FFT result.
function M.rfft2(x)
  return native.rfft2(x)
end

--- Inverse of rfft2 (2-D complex to real).
-- @tparam Array x Complex input from rfft2.
-- @treturn Array Real-valued 2-D IFFT result.
function M.irfft2(x)
  return native.irfft2(x)
end

--- N-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @treturn Array Complex N-D FFT result.
function M.rfftn(x)
  return native.rfftn(x)
end

--- Inverse of rfftn (N-D complex to real).
-- @tparam Array x Complex input from rfftn.
-- @treturn Array Real-valued N-D IFFT result.
function M.irfftn(x)
  return native.irfftn(x)
end

return M
