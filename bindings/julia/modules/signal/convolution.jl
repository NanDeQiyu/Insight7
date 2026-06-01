# modules/signal/convolution.jl
# Convolution and correlation functions.

"""
    fftconvolve(in1::InsightArray, in2::InsightArray; mode::String="full") -> InsightArray

Convolve two N-D arrays using FFT.

# Arguments
- `in1::InsightArray`: First input array.
- `in2::InsightArray`: Second input array.
- `mode::String`: Convolution mode (`"full"`, `"same"`, `"valid"`).
"""
function fftconvolve end

"""
    correlate(in1::InsightArray, in2::InsightArray; mode::String="full") -> InsightArray

Cross-correlate two N-D arrays.
"""
function correlate end

"""
    correlation_lags(in1_len::Int, in2_len::Int; mode::String="full") -> InsightArray

Return the lag values for cross-correlation.
"""
function correlation_lags end

"""
    convolve2d(in1::InsightArray, in2::InsightArray; mode::String="full") -> InsightArray

Convolve two 2-D arrays.
"""
function convolve2d end

"""
    correlate2d(in1::InsightArray, in2::InsightArray; mode::String="full") -> InsightArray

Cross-correlate two 2-D arrays.
"""
function correlate2d end
