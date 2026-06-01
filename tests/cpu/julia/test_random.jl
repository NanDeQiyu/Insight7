# Random number generation CPU binding tests
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_random.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0
failed = 0

function check(name, cond)
    global passed, failed
    if cond
        passed += 1
    else
        failed += 1
        println("FAIL: $name")
    end
end

println("=== Random ===")

# rand shape
a = Insight.rand(Int64[3, 4], Insight.float64)
check("rand_shape", Insight.numel(a) == 12)

# randn shape
a = Insight.randn(Int64[3, 4], Insight.float64)
check("randn_shape", Insight.numel(a) == 12)

# seed determinism
Insight.seed(UInt64(42))
a = Insight.rand(Int64[5], Insight.float64)
Insight.seed(UInt64(42))
b = Insight.rand(Int64[5], Insight.float64)
d_a = Insight.to_array(a)
d_b = Insight.to_array(b)
check("seed_determinism", d_a == d_b)

# get_seed
Insight.seed(UInt64(12345))
s = Insight.get_seed()
check("get_seed", s == UInt64(12345))

# rand_like
template = Insight.zeros(Int64[3, 4], Insight.float64)
rl = Insight.rand_like(template)
check("rand_like", Insight.numel(rl) == 12)

# randn_like
rn = Insight.randn_like(template)
check("randn_like", Insight.numel(rn) == 12)

# exponential
e = Insight.exponential(1.0, Int64[100], dtype_val=Insight.float64)
check("exponential", Insight.numel(e) == 100)

# gamma
g = Insight.gamma_dist(2.0, 1.0, Int64[100], dtype_val=Insight.float64)
check("gamma", Insight.numel(g) == 100)

# beta
b = Insight.beta_dist(2.0, 5.0, Int64[100], dtype_val=Insight.float64)
check("beta", Insight.numel(b) == 100)

# binomial
bi = Insight.binomial_dist(Int64(10), 0.5, Int64[100], dtype_val=Insight.int64)
check("binomial", Insight.numel(bi) == 100)

# poisson
p = Insight.poisson_dist(5.0, Int64[100], dtype_val=Insight.int64)
check("poisson", Insight.numel(p) == 100)

# rand 1d
a1 = Insight.rand(Int64[10], Insight.float32)
check("rand_1d", Insight.numel(a1) == 10)

# rand 3d
a3 = Insight.rand(Int64[2, 3, 4], Insight.float64)
check("rand_3d", Insight.numel(a3) == 24)

# randn 1d
n1 = Insight.randn(Int64[20], Insight.float64)
check("randn_1d", Insight.numel(n1) == 20)

# exponential with different scale
e2 = Insight.exponential(2.0, Int64[50], dtype_val=Insight.float64)
check("exponential_scale", Insight.numel(e2) == 50)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
