# modules/signal/acoustics.jl
# Acoustic scale conversion functions.

"""
    mel2hz(mel::InsightArray) -> InsightArray

Convert mel frequencies to Hz.
"""
function mel2hz end

"""
    hz2mel(hz::InsightArray) -> InsightArray

Convert Hz frequencies to mel scale.
"""
function hz2mel end

"""
    mel_frequencies(num_mels::Int; fmin::Float64=0.0, fmax::Float64=11025.0) -> InsightArray

Return an array of mel-spaced frequencies.
"""
function mel_frequencies end

"""
    hz2bark(hz::InsightArray) -> InsightArray

Convert Hz frequencies to Bark scale.
"""
function hz2bark end

"""
    bark2hz(bark::InsightArray) -> InsightArray

Convert Bark scale frequencies to Hz.
"""
function bark2hz end
