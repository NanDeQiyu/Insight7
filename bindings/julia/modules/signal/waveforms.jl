# modules/signal/waveforms.jl
# Waveform generation functions.

"""
    sawtooth(t::InsightArray, width::Float64=1.0) -> InsightArray

Return a periodic sawtooth waveform.

# Arguments
- `t::InsightArray`: Time array.
- `width::Float64`: Width of the rising ramp (0 to 1). Default 1.0.
"""
function sawtooth end

"""
    square_wf(t::InsightArray, duty::Float64=0.5) -> InsightArray

Return a periodic square-wave waveform.

# Arguments
- `t::InsightArray`: Time array.
- `duty::Float64`: Duty cycle (0 to 1). Default 0.5.
"""
function square_wf end

"""
    chirp(t::InsightArray, f0::Float64, t1::Float64, f1::Float64;
          method::String="linear", phi::Float64=0.0) -> InsightArray

Return a swept-frequency cosine (chirp) signal.

# Arguments
- `t::InsightArray`: Time array.
- `f0::Float64`: Frequency at time 0.
- `t1::Float64`: Time at which `f1` is specified.
- `f1::Float64`: Frequency at time `t1`.
- `method::String`: Chirp method (`"linear"`, `"quadratic"`, `"logarithmic"`, `"hyperbolic"`).
- `phi::Float64`: Phase offset in degrees.
"""
function chirp end

"""
    unit_impulse(shape::Vector{Int64}; idx::String="mid") -> InsightArray

Return an impulse (delta) function.

# Arguments
- `shape::Vector{Int64}`: Shape of the output array.
- `idx::String`: Position of the impulse (`"mid"`, `"begin"`, `"end"`).
"""
function unit_impulse end
