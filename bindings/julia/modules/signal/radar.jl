# modules/signal/radar.jl
# Radar signal processing functions.

"""
    cfar_alpha(pfa::Float64, N::Int) -> Float64

Compute the CA-CFAR threshold multiplier for given PFA and window size.
"""
function cfar_alpha(pfa::Float64, N::Int)::Float64
    return ccall((:insight_jl_cfar_alpha, LIB_INSIGHT), Float64,
                 (Float64, Int32), pfa, Int32(N))
end

"""
    pulse_compression(x::InsightArray, tpl::InsightArray; normalize=false) -> InsightArray

Perform pulse compression (matched filtering) using FFT correlation.
"""
function pulse_compression(x::InsightArray, tpl::InsightArray;
                           normalize::Bool=false)::InsightArray
    ptr = ccall((:insight_jl_pulse_compression, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}, Int32), x, tpl, Int32(normalize ? 1 : 0))
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    pulse_doppler(x::InsightArray) -> InsightArray

Compute the pulse-Doppler processing (FFT along slow-time axis).
"""
function pulse_doppler(x::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_pulse_doppler, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid},), x)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end

"""
    mvdr(x::InsightArray, sv::InsightArray) -> InsightArray

Apply MVDR (Capon) beamforming.
"""
function mvdr(x::InsightArray, sv::InsightArray)::InsightArray
    ptr = ccall((:insight_jl_mvdr, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), x, sv)
    arr = InsightArray(ptr); finalizer(_free, arr); return arr
end
