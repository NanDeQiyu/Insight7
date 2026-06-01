# modules/signal/spectral.jl
# Spectral analysis functions.

"""
    welch_jl(x::InsightArray; fs::Float64=1.0, nperseg::Int=256,
             noverlap::Int=0, nfft::Int=-1) -> (frequencies, power)

Estimate power spectral density using Welch's method.
"""
function welch_jl end

"""
    periodogram_jl(x::InsightArray; fs::Float64=1.0, nfft::Int=-1) -> (frequencies, power)

Compute the power spectral density using a periodogram.
"""
function periodogram_jl end

"""
    csd(x::InsightArray, y::InsightArray; fs::Float64=1.0, nperseg::Int=256,
        noverlap::Int=0) -> (frequencies, power)

Compute the cross-spectral density using Welch's method.
"""
function csd end

"""
    coherence(x::InsightArray, y::InsightArray; fs::Float64=1.0, nperseg::Int=256,
              noverlap::Int=0) -> (frequencies, coherence)

Compute the magnitude-squared coherence estimate.
"""
function coherence end

"""
    spectrogram(x::InsightArray; fs::Float64=1.0, nperseg::Int=256,
                noverlap::Int=0, nfft::Int=-1) -> (frequencies, times, Sxx)

Compute a spectrogram using short-time FFT.
"""
function spectrogram end

"""
    stft(x::InsightArray; fs::Float64=1.0, nperseg::Int=256,
         noverlap::Int=0, nfft::Int=-1) -> (frequencies, times, Zxx)

Compute the Short-Time Fourier Transform.
"""
function stft end

"""
    vectorstrength(events::InsightArray, period::Float64) -> (strength, phase)

Compute the vector strength of events relative to a period.
"""
function vectorstrength end

"""
    choose_conv_method(in1::InsightArray, in2::InsightArray; mode::String="full") -> String

Choose the faster convolution method (direct or FFT) for the given inputs.
"""
function choose_conv_method end

"""
    firfilter_zi_state(b::InsightArray, x::InsightArray, zi::InsightArray; axis::Int=-1) -> (y, zf)

Apply an FIR filter with initial/return final state.
"""
function firfilter_zi_state end
