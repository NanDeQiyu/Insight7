--- Fast Fourier Transform operations.
--
-- Provides 1-D, 2-D, and N-D FFT and inverse FFT, real-input FFT,
-- sample frequency generation, and related utilities.
--
-- @module insight.fft

local native = require("_insight")
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

--- 1-D discrete Fourier transform.
-- @tparam Array x Input array.
-- @int[opt=-1] n FFT length. -1 uses input length.
-- @int[opt=-1] axis Axis along which to compute. -1 is last axis.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array FFT result.
M.fft = _wrap({ "x", "n", "axis", "norm" }, function(x, n, axis, norm)
  return native.fft(x, n or -1, axis or -1, norm or "backward")
end)

--- 1-D inverse discrete Fourier transform.
-- @tparam Array x Input frequency-domain array.
-- @int[opt=-1] n IFFT length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array IFFT result.
M.ifft = _wrap({ "x", "n", "axis", "norm" }, function(x, n, axis, norm)
  return native.ifft(x, n or -1, axis or -1, norm or "backward")
end)

--- 1-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @int[opt=-1] n FFT length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array Complex FFT result (positive frequencies only).
M.rfft = _wrap({ "x", "n", "axis", "norm" }, function(x, n, axis, norm)
  return native.rfft(x, n or -1, axis or -1, norm or "backward")
end)

--- Inverse of rfft (complex to real).
-- @tparam Array x Complex input from rfft.
-- @int[opt=-1] n Output length.
-- @int[opt=-1] axis Axis along which to compute.
-- @string[opt="backward"] norm Normalization mode.
-- @treturn Array Real-valued IFFT result.
M.irfft = _wrap({ "x", "n", "axis", "norm" }, function(x, n, axis, norm)
  return native.irfft(x, n or -1, axis or -1, norm or "backward")
end)

--- DFT sample frequencies.
-- @int n Window length.
-- @number[opt=1] d Sample spacing.
-- @treturn Array Frequency vector.
M.fftfreq = _wrap({ "n", "d" }, function(n, d)
  return native.fftfreq(n, d or 1.0)
end)

--- DFT sample frequencies for rfft.
-- @int n Window length.
-- @number[opt=1] d Sample spacing.
-- @treturn Array Frequency vector (non-negative only).
M.rfftfreq = _wrap({ "n", "d" }, function(n, d)
  return native.rfftfreq(n, d or 1.0)
end)

--- Shift zero-frequency component to center of spectrum.
-- @tparam Array x Input frequency-domain array.
-- @treturn Array Shifted array.
M.fftshift = _wrap({ "x" }, function(x)
  return native.fftshift(x)
end)

--- Inverse of fftshift.
-- @tparam Array x Input shifted array.
-- @treturn Array Un-shifted array.
M.ifftshift = _wrap({ "x" }, function(x)
  return native.ifftshift(x)
end)

--- Find the next fast FFT length.
-- @int target Target length.
-- @bool[opt=true] real If true, consider real FFT efficiency.
-- @int Fast FFT length >= target.
M.next_fast_len = _wrap({ "target", "real" }, function(target, real)
  return native.next_fast_len(target, real == nil and true or real)
end)

--- FFT of a Hermitian signal (real output).
-- @tparam Array x Hermitian-symmetric input.
-- @int[opt=-1] n Output length.
-- @treturn Array Real-valued FFT result.
M.hfft = _wrap({ "x", "n" }, function(x, n)
  return native.hfft(x, n or -1)
end)

--- Inverse of hfft.
-- @tparam Array x Real input.
-- @int[opt=-1] n Output length.
-- @treturn Array Hermitian-symmetric output.
M.ihfft = _wrap({ "x", "n" }, function(x, n)
  return native.ihfft(x, n or -1)
end)

--- 2-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @treturn Array Complex 2-D FFT result.
M.rfft2 = _wrap({ "x" }, function(x)
  return native.rfft2(x)
end)

--- Inverse of rfft2 (2-D complex to real).
-- @tparam Array x Complex input from rfft2.
-- @treturn Array Real-valued 2-D IFFT result.
M.irfft2 = _wrap({ "x" }, function(x)
  return native.irfft2(x)
end)

--- N-D real-input discrete Fourier transform.
-- @tparam Array x Real input array.
-- @treturn Array Complex N-D FFT result.
M.rfftn = _wrap({ "x" }, function(x)
  return native.rfftn(x)
end)

--- Inverse of rfftn (N-D complex to real).
-- @tparam Array x Complex input from rfftn.
-- @treturn Array Real-valued N-D IFFT result.
M.irfftn = _wrap({ "x" }, function(x)
  return native.irfftn(x)
end)

return M
