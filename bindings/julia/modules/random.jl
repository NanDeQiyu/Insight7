# modules/random.jl
# Random number generation functions.

"""
    rand(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Generate an array of uniform random numbers in [0, 1).
"""
function rand end

"""
    randn(dims::Vector{Int64}, dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Generate an array of standard normal random numbers.
"""
function randn end

"""
    seed(s::Integer) -> Nothing

Set the random number generator seed.
"""
function seed end

"""
    get_seed() -> UInt64

Get the current random number generator seed.
"""
function get_seed end

"""
    rand_like(x::InsightArray) -> InsightArray

Generate uniform random array with the same shape and dtype as `x`.
"""
function rand_like end

"""
    randn_like(x::InsightArray) -> InsightArray

Generate standard normal random array with the same shape and dtype as `x`.
"""
function randn_like end

"""
    exponential(scale::Float64, dims::Vector{Int64};
                dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Generate an array of exponential random numbers.
"""
function exponential end

"""
    gamma_dist(shape_param::Float64, rate::Float64, dims::Vector{Int64};
               dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Generate an array of gamma-distributed random numbers.
"""
function gamma_dist end

"""
    beta_dist(a::Float64, b::Float64, dims::Vector{Int64};
             dtype_val::Int32=float32, device::Int32=CPU) -> InsightArray

Generate an array of beta-distributed random numbers.
"""
function beta_dist end

"""
    binomial_dist(n::Int64, p::Float64, dims::Vector{Int64};
                  dtype_val::Int32=int64, device::Int32=CPU) -> InsightArray

Generate an array of binomial-distributed random numbers.
"""
function binomial_dist end

"""
    poisson_dist(lam::Float64, dims::Vector{Int64};
                 dtype_val::Int32=int64, device::Int32=CPU) -> InsightArray

Generate an array of Poisson-distributed random numbers.
"""
function poisson_dist end
