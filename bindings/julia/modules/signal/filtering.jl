# modules/signal/filtering.jl
# Digital filtering functions.

"""
    hilbert(x::InsightArray; N::Int=-1) -> InsightArray

Compute the analytic signal using the Hilbert transform.
"""
function hilbert end

"""
    hilbert2(x::InsightArray; N::Int=-1) -> InsightArray

Compute the 2-D analytic signal using the Hilbert transform.
"""
function hilbert2 end

"""
    detrend(data::InsightArray; axis::Int=-1, type::String="linear") -> InsightArray

Remove a trend from the data along the given axis.
"""
function detrend end

"""
    wiener(im::InsightArray; mysize::Vector{Int64}=Int64[], noise::Float64=-1.0) -> InsightArray

Apply a Wiener filter to an N-dimensional array.
"""
function wiener end

"""
    firfilter(b::InsightArray, x::InsightArray; axis::Int=-1) -> InsightArray

Apply an FIR filter to a signal.
"""
function firfilter end

"""
    lfilter(b::InsightArray, a::InsightArray, x::InsightArray; axis::Int=-1) -> InsightArray

Apply an IIR or FIR filter to a signal using direct-form II.
"""
function lfilter end

"""
    lfilter_zi(b::InsightArray, a::InsightArray) -> InsightArray

Compute the initial state for `lfilter` for step-response steady-state.
"""
function lfilter_zi end

"""
    filtfilt(b::InsightArray, a::InsightArray, x::InsightArray; axis::Int=-1) -> InsightArray

Apply a forward-backward digital filter (zero-phase filtering).
"""
function filtfilt end

"""
    decimate(x::InsightArray, q::Int; axis::Int=-1, zero_phase::Bool=true) -> InsightArray

Downsample a signal after applying an anti-aliasing filter.
"""
function decimate end

"""
    resample(x::InsightArray, num::Int; axis::Int=-1) -> InsightArray

Resample a signal to a given number of samples using FFT.
"""
function resample end

"""
    resample_poly(x::InsightArray, up::Int, down::Int; axis::Int=-1) -> InsightArray

Resample a signal using polyphase filtering.
"""
function resample_poly end

"""
    freq_shift(x::InsightArray, freq::Float64, fs::Float64) -> InsightArray

Shift the frequency content of a signal.
"""
function freq_shift end
