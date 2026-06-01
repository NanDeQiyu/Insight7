# modules/signal/radar.jl
# Radar signal processing functions.

"""
    cfar_alpha(pfa::Float64, N::Int) -> Float64

Compute the CA-CFAR threshold multiplier for a given probability of false alarm.

# Arguments
- `pfa::Float64`: Probability of false alarm.
- `N::Int`: Number of reference cells.

# Returns
- `Float64`: CFAR threshold multiplier (alpha).
"""
function cfar_alpha end

"""
    pulse_compression(signal::InsightArray, matched_filter::InsightArray) -> InsightArray

Perform pulse compression using a matched filter.
"""
function pulse_compression end

"""
    pulse_doppler(cpi::InsightArray, prf::Float64) -> InsightArray

Perform pulse-Doppler processing on a coherent processing interval.
"""
function pulse_doppler end

"""
    mvdr(x::InsightArray, sv::InsightArray; reg::Float64=1e-6) -> InsightArray

Compute MVDR (Capon) beamformer output power.
"""
function mvdr end
