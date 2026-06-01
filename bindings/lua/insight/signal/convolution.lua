--- Convolution and correlation functions.
-- Provides FFT-based convolution, cross-correlation,
-- and 2-D convolution/correlation operations.
-- @module insight.signal.convolution

local native = require("_insight")
local sig = native.signal

local M = {}

--- Convolve two N-dimensional arrays using FFT.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string[opt="full"] mode Convolution mode: 'full', 'valid', or 'same'.
-- @treturn Array Convolved array.
function M.fftconvolve(a, b, mode)
  return sig.fftconvolve(a, b, mode or "full")
end

--- Cross-correlate two N-dimensional arrays.
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string[opt="full"] mode Correlation mode: 'full', 'valid', or 'same'.
-- @treturn Array Cross-correlation result.
function M.correlate(a, b, mode)
  return sig.correlate(a, b, mode or "full")
end

--- 2-D convolution of two arrays.
-- @tparam Array a First input 2-D array.
-- @tparam Array b Second input 2-D array.
-- @string[opt="full"] mode Convolution mode: 'full', 'valid', or 'same'.
-- @string[opt="fill"] boundary Boundary handling: 'fill', 'wrap', 'symm'.
-- @number[opt=0] fillvalue Value for fill boundary.
-- @treturn Array 2-D convolved array.
function M.convolve2d(a, b, mode, boundary, fillvalue)
  return sig.convolve2d(a, b, mode or "full", boundary or "fill", fillvalue or 0)
end

--- 2-D cross-correlation of two arrays.
-- @tparam Array a First input 2-D array.
-- @tparam Array b Second input 2-D array.
-- @string[opt="full"] mode Correlation mode: 'full', 'valid', or 'same'.
-- @string[opt="fill"] boundary Boundary handling: 'fill', 'wrap', 'symm'.
-- @number[opt=0] fillvalue Value for fill boundary.
-- @treturn Array 2-D cross-correlation result.
function M.correlate2d(a, b, mode, boundary, fillvalue)
  return sig.correlate2d(a, b, mode or "full", boundary or "fill", fillvalue or 0)
end

--- Choose the best convolution method (direct vs FFT).
-- @tparam Array a First input array.
-- @tparam Array b Second input array.
-- @string mode Convolution mode.
-- @treturn string Best method: 'direct' or 'fft'.
function M.choose_conv_method(a, b, mode)
  return sig.choose_conv_method(a, b, mode)
end

--- Compute the lag values for cross-correlation.
-- @int input_length Length of the first input.
-- @int kernel_length Length of the second input (kernel).
-- @string[opt="full"] mode Correlation mode.
-- @treturn Array Lag values.
function M.correlation_lags(input_length, kernel_length, mode)
  return sig.correlation_lags(input_length, kernel_length, mode or "full")
end

return M
