# modules/signal/demod.jl
# Demodulation functions.

"""
    fm_demod(x::InsightArray; fs::Float64=1.0) -> InsightArray

Demodulate an FM signal by computing the instantaneous frequency.

# Arguments
- `x::InsightArray`: Input FM signal.
- `fs::Float64`: Sampling frequency. Default 1.0.

# Returns
- `InsightArray`: Demodulated signal.
"""
function fm_demod end
