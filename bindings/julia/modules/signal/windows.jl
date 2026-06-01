# modules/signal/windows.jl
# Window functions for signal processing.

"""
    hann(M::Int; sym::Bool=true) -> InsightArray

Return a Hann window of length `M`.
"""
function hann(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_hann, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    hamming(M::Int; sym::Bool=true) -> InsightArray

Return a Hamming window of length `M`.
"""
function hamming(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_hamming, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    blackman(M::Int; sym::Bool=true) -> InsightArray

Return a Blackman window of length `M`.
"""
function blackman(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_blackman, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    kaiser(M::Int, beta::Float64; sym::Bool=true) -> InsightArray

Return a Kaiser window of length `M` with shape parameter `beta`.
"""
function kaiser(M::Int, beta::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_kaiser, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), beta, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    gaussian(M::Int, std_val::Float64; sym::Bool=true) -> InsightArray

Return a Gaussian window of length `M` with standard deviation `std_val`.
"""
function gaussian(M::Int, std_val::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_gaussian, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), std_val, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    boxcar(M::Int; sym::Bool=true) -> InsightArray

Return a boxcar (rectangular) window of length `M`.
"""
function boxcar(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_boxcar, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    triang(M::Int; sym::Bool=true) -> InsightArray

Return a triangular window of length `M`.
"""
function triang(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_triang, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    bartlett(M::Int; sym::Bool=true) -> InsightArray

Return a Bartlett window of length `M`.
"""
function bartlett(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_bartlett, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    flattop(M::Int; sym::Bool=true) -> InsightArray

Return a flat-top window of length `M`.
"""
function flattop(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_flattop, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    nuttall(M::Int; sym::Bool=true) -> InsightArray

Return a Nuttall minimum 4-term Blackman-Harris window of length `M`.
"""
function nuttall(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_nuttall, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    blackmanharris(M::Int; sym::Bool=true) -> InsightArray

Return a Blackman-Harris window of length `M`.
"""
function blackmanharris(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_blackmanharris, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    tukey(M::Int, alpha::Float64=0.5; sym::Bool=true) -> InsightArray

Return a Tukey (tapered cosine) window of length `M`.
"""
function tukey(M::Int; alpha::Float64=0.5, sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_tukey, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), alpha, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    chebwin(M::Int, at::Float64; sym::Bool=true) -> InsightArray

Return a Dolph-Chebyshev window of length `M` with attenuation `at` dB.
"""
function chebwin(M::Int, at::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_chebwin, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), at, sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    taylor(M::Int; nbar=4, sll=-30.0, norm=true, sym=true) -> InsightArray

Return a Taylor window of length `M`.
"""
function taylor(M::Int; nbar::Int=4, sll::Float64=-30.0,
                norm::Bool=true, sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_taylor, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int64, Float64, Int32, Int32),
                Int64(M), Int64(nbar), sll, norm ? Int32(1) : Int32(0),
                sym ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    cosine_win(M::Int; sym::Bool=true) -> InsightArray

Return a cosine window of length `M`.
"""
function cosine_win(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_cosine_win, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    general_hamming(M::Int, alpha::Float64; sym::Bool=true) -> InsightArray

Return a generalized Hamming window of length `M`.
"""
function general_hamming(M::Int, alpha::Float64; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_general_hamming, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Int32), Int64(M), alpha, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    parzen_win(M::Int; sym::Bool=true) -> InsightArray

Return a Parzen window of length `M`.
"""
function parzen_win(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_parzen, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    bohman_win(M::Int; sym::Bool=true) -> InsightArray

Return a Bohman window of length `M`.
"""
function bohman_win(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_bohman, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    barthann_win(M::Int; sym::Bool=true) -> InsightArray

Return a Bartlett-Hann window of length `M`.
"""
function barthann_win(M::Int; sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_barthann, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Int32), Int64(M), Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    exponential_win(M::Int; center=-1.0, tau=1.0, sym=true) -> InsightArray

Return an exponential (Poisson) window of length `M`.
"""
function exponential_win(M::Int; center::Float64=-1.0, tau::Float64=1.0,
                         sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_exponential_win, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32), Int64(M), center, tau, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    general_gaussian_win(M::Int, p::Float64, sig::Float64; sym=true) -> InsightArray

Return a generalized Gaussian window of length `M`.
"""
function general_gaussian_win(M::Int, p::Float64, sig::Float64;
                              sym::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_general_gaussian, LIB_INSIGHT), Ptr{Cvoid},
                (Int64, Float64, Float64, Int32), Int64(M), p, sig, Int32(sym ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    get_window(window::String, Nx::Int; fftbins::Bool=true) -> InsightArray

Return a window of the given type and length.
"""
function get_window(window::String, Nx::Int; fftbins::Bool=true)::InsightArray
    ptr = ccall((:insight_jl_get_window, LIB_INSIGHT), Ptr{Cvoid},
                (Cstring, Int64, Int32), window, Int64(Nx),
                fftbins ? Int32(1) : Int32(0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
