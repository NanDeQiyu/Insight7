# modules/signal/windows.jl
# Window functions for signal processing.
# These are documentation wrappers; the actual implementations live in Insight.jl.

"""
    hann(M::Int; sym::Bool=true) -> InsightArray

Return a Hann window of length `M`.

# Arguments
- `M::Int`: Number of points in the output window.
- `sym::Bool`: If `true` (default), generates a symmetric window.
"""
function hann end

"""
    hamming(M::Int; sym::Bool=true) -> InsightArray

Return a Hamming window of length `M`.
"""
function hamming end

"""
    blackman(M::Int; sym::Bool=true) -> InsightArray

Return a Blackman window of length `M`.
"""
function blackman end

"""
    kaiser(M::Int, beta::Float64; sym::Bool=true) -> InsightArray

Return a Kaiser window of length `M` with shape parameter `beta`.
"""
function kaiser end

"""
    gaussian(M::Int, std_val::Float64; sym::Bool=true) -> InsightArray

Return a Gaussian window of length `M` with standard deviation `std_val`.
"""
function gaussian end

"""
    boxcar(M::Int; sym::Bool=true) -> InsightArray

Return a boxcar (rectangular) window of length `M`.
"""
function boxcar end

"""
    triang(M::Int; sym::Bool=true) -> InsightArray

Return a triangular window of length `M`.
"""
function triang end

"""
    bartlett(M::Int; sym::Bool=true) -> InsightArray

Return a Bartlett window of length `M`.
"""
function bartlett end

"""
    flattop(M::Int; sym::Bool=true) -> InsightArray

Return a flat-top window of length `M`.
"""
function flattop end

"""
    nuttall(M::Int; sym::Bool=true) -> InsightArray

Return a Nuttall minimum 4-term Blackman-Harris window of length `M`.
"""
function nuttall end

"""
    blackmanharris(M::Int; sym::Bool=true) -> InsightArray

Return a Blackman-Harris window of length `M`.
"""
function blackmanharris end

"""
    tukey(M::Int; alpha::Float64=0.5, sym::Bool=true) -> InsightArray

Return a Tukey (tapered cosine) window of length `M`.
"""
function tukey end

"""
    chebwin(M::Int, at::Float64; sym::Bool=true) -> InsightArray

Return a Dolph-Chebyshev window of length `M` with attenuation `at` dB.
"""
function chebwin end

"""
    taylor(M::Int; nbar::Int=4, sll::Float64=-30.0, norm::Bool=true, sym::Bool=true) -> InsightArray

Return a Taylor window of length `M`.
"""
function taylor end

"""
    get_window(window::String, Nx::Int; fftbins::Bool=true) -> InsightArray

Return a window of the given type and length.

# Arguments
- `window::String`: Window type name (e.g. `"hann"`, `"hamming"`, `"kaiser"`).
- `Nx::Int`: Number of points in the output window.
- `fftfbins::Bool`: If `true`, create a periodic window for FFT use.
"""
function get_window end

"""
    cosine_win(M::Int; sym::Bool=true) -> InsightArray

Return a cosine window of length `M`.
"""
function cosine_win end

"""
    general_hamming(M::Int, alpha::Float64; sym::Bool=true) -> InsightArray

Return a generalized Hamming window of length `M` with coefficient `alpha`.
"""
function general_hamming end

"""
    parzen_win(M::Int; sym::Bool=true) -> InsightArray

Return a Parzen window of length `M`.
"""
function parzen_win end

"""
    bohman_win(M::Int; sym::Bool=true) -> InsightArray

Return a Bohman window of length `M`.
"""
function bohman_win end

"""
    barthann_win(M::Int; sym::Bool=true) -> InsightArray

Return a Bartlett-Hann window of length `M`.
"""
function barthann_win end

"""
    exponential_win(M::Int; tau::Float64=1.0; sym::Bool=true) -> InsightArray

Return an exponential (Poisson) window of length `M`.
"""
function exponential_win end

"""
    general_gaussian_win(M::Int, p::Float64, sig::Float64; sym::Bool=true) -> InsightArray

Return a generalized Gaussian window of length `M`.
"""
function general_gaussian_win end
