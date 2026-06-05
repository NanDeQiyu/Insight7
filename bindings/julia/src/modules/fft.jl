# modules/fft.jl
# Fast Fourier Transform functions.

"""
    fft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

Compute the 1-D discrete Fourier transform.

# Arguments
- `x::InsightArray`: Input array.
- `n::Int`: FFT length. If `nothing`, uses the length of `x`.
"""
function fft end

"""
    ifft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

Compute the 1-D inverse discrete Fourier transform.
"""
function ifft end

"""
    rfft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

Compute the 1-D DFT of a real input (returns complex, half-spectrum).
"""
function rfft end

"""
    fftfreq(n::Int; d::Float64=1.0) -> InsightArray

Return the DFT sample frequencies for a signal of length `n`.
"""
function fftfreq end

"""
    fftshift(x::InsightArray; axis::Int=-1) -> InsightArray

Shift the zero-frequency component to the center of the spectrum.
"""
function fftshift end

"""
    ifftshift(x::InsightArray; axis::Int=-1) -> InsightArray

Inverse of `fftshift`.
"""
function ifftshift end

"""
    next_fast_len(target::Int) -> Int

Return the next fast FFT length >= `target`.
"""
function next_fast_len end

"""
    hfft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

Compute the FFT of a Hermitian-symmetric signal (real output).
"""
function hfft end

"""
    ihfft(x::InsightArray; n::Union{Int,Nothing}=nothing) -> InsightArray

Inverse of `hfft`.
"""
function ihfft end

"""
    rfft2(x::InsightArray; s::Vector{Int64}=Int64[], axes::Vector{Int32}=Int32[-2, -1]) -> InsightArray

Compute the 2-D DFT of a real input.
"""
function rfft2 end

"""
    irfft2(x::InsightArray; s::Vector{Int64}=Int64[], axes::Vector{Int32}=Int32[-2, -1]) -> InsightArray

Inverse of `rfft2`.
"""
function irfft2 end

"""
    rfftn(x::InsightArray; s::Vector{Int64}=Int64[], axes::Vector{Int32}=Int32[]) -> InsightArray

Compute the N-D DFT of a real input.
"""
function rfftn end

"""
    irfftn(x::InsightArray; s::Vector{Int64}=Int64[], axes::Vector{Int32}=Int32[]) -> InsightArray

Inverse of `rfftn`.
"""
function irfftn end
