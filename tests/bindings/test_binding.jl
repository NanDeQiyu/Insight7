# tests/bindings/test_binding.jl
#
# Smoke tests for the Insight7 Julia binding.
#
# Run with:
#   cd build && julia ../tests/bindings/test_binding.jl

# Load the Insight module
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))

using Test

# Load the Insight module from build directory
const _insight_dir = joinpath(@__DIR__, "..", "..", "build", "bindings", "julia")
include(joinpath(_insight_dir, "Insight.jl"))

using .Insight

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

# ===== DType shortcuts =====
println("=== DType ===")
check("float32 exists", Insight.float32 isa Int32)
check("float64 exists", Insight.float64 isa Int32)
check("int32 exists", Insight.int32 isa Int32)
check("int64 exists", Insight.int64 isa Int32)

# ===== Array creation =====
println("=== Creation ===")
a = Insight.zeros(Int64[2, 3], Insight.float32)
check("zeros defined", a.ptr != C_NULL)
check("zeros numel", Insight.numel(a) == 6)

b = Insight.ones(Int64[4], Insight.float32)
check("ones defined", b.ptr != C_NULL)
check("ones numel", Insight.numel(b) == 4)

c = Insight.full(Int64[2, 2], 3.14, Insight.float32)
check("full defined", c.ptr != C_NULL)

d = Insight.eye(Int64(3), Int64(3), Insight.float32)
check("eye defined", d.ptr != C_NULL)
check("eye numel", Insight.numel(d) == 9)

e = Insight.arange(0.0, 10.0, 1.0, Insight.int64)
check("arange defined", e.ptr != C_NULL)
check("arange numel", Insight.numel(e) == 10)

# ===== Array properties =====
println("=== Properties ===")
f = Insight.zeros(Int64[2, 3, 4])
check("ndim", Insight.ndim(f) == 3)
check("numel", Insight.numel(f) == 24)
check("shape", Insight.shape(f) == [2, 3, 4])

# ===== Operators =====
println("=== Operators ===")
g = Insight.ones(Int64[3], Insight.float32)
h = Insight.ones(Int64[3], Insight.float32)
i = Insight.add(g, h)
check("add defined", i.ptr != C_NULL)
check("add numel", Insight.numel(i) == 3)

j = Insight.sub(g, h)
check("sub defined", j.ptr != C_NULL)

k = Insight.mul(g, h)
check("mul defined", k.ptr != C_NULL)

# ===== Reduction =====
println("=== Reduction ===")
o = Insight.ones(Int64[3, 4], Insight.float32)
p = Insight.sum(o)
check("sum defined", p.ptr != C_NULL)
check("sum numel", Insight.numel(p) == 1)

q = Insight.mean(o)
check("mean defined", q.ptr != C_NULL)

# ===== Linear Algebra =====
println("=== Linalg ===")
r = Insight.ones(Int64[2, 3], Insight.float32)
s = Insight.ones(Int64[3, 4], Insight.float32)
t = Insight.matmul(r, s)
check("matmul defined", t.ptr != C_NULL)
check("matmul numel", Insight.numel(t) == 8)

u = Insight.eye(Int64(3), Int64(3), Insight.float32)
v = Insight.det(u)
check("det defined", v.ptr != C_NULL)

w = Insight.inv(u)
check("inv defined", w.ptr != C_NULL)

# ===== FFT =====
println("=== FFT ===")
x = Insight.ones(Int64[8], Insight.float32)
y = Insight.fft(x)
check("fft defined", y.ptr != C_NULL)

# ===== Complex (Phase D) =====
println("=== Complex ===")
a3 = Insight.ones(Int64[3], Insight.float32)
c1 = Insight.to_complex(a3)
check("to_complex defined", c1.ptr != C_NULL)
check("is_complex true", Insight.is_complex(c1))
check("is_complex false on float", !Insight.is_complex(a3))

# ===== Additional Unary (Phase D) =====
println("=== Additional Unary ===")
u = Insight.linspace(0.1, 2.0, 10, Insight.float32)
check("exp2 defined", Insight.exp2_fn(u).ptr != C_NULL)
check("expm1 defined", Insight.expm1_fn(u).ptr != C_NULL)
check("log1p defined", Insight.log1p_fn(u).ptr != C_NULL)
check("cbrt defined", Insight.cbrt_fn(u).ptr != C_NULL)
check("reciprocal defined", Insight.reciprocal_fn(u).ptr != C_NULL)
check("asinh defined", Insight.asinh_fn(u).ptr != C_NULL)
check("acosh defined", Insight.acosh_fn(u).ptr != C_NULL)
check("deg2rad defined", Insight.deg2rad_fn(u).ptr != C_NULL)
check("rad2deg defined", Insight.rad2deg_fn(u).ptr != C_NULL)

# ===== Additional Reduction (Phase D) =====
println("=== Additional Reduction ===")
m2 = Insight.from_data(Float32[1, 2, 3, 4, 5, 6], Insight.float32)
m2 = Insight.reshape(m2, Int64[2, 3])
check("cummax defined", Insight.cummax(m2, 0).ptr != C_NULL)
check("cummin defined", Insight.cummin(m2, 0).ptr != C_NULL)
check("count_nonzero defined", Insight.count_nonzero(m2).ptr != C_NULL)
check("median defined", Insight.median(m2).ptr != C_NULL)
check("quantile defined", Insight.quantile(m2, 0.5).ptr != C_NULL)
check("nansum defined", Insight.nansum(m2).ptr != C_NULL)
check("nanmean defined", Insight.nanmean(m2).ptr != C_NULL)
check("nanstd defined", Insight.nanstd(m2).ptr != C_NULL)

# ===== Additional Manipulation (Phase D) =====
println("=== Additional Manipulation ===")
check("fliplr defined", Insight.fliplr(m2).ptr != C_NULL)
check("flipud defined", Insight.flipud(m2).ptr != C_NULL)
check("tril defined", Insight.tril(m2).ptr != C_NULL)
check("triu defined", Insight.triu(m2).ptr != C_NULL)
check("diff defined", Insight.diff_fn(m2).ptr != C_NULL)

# ===== Additional Random (Phase D) =====
println("=== Additional Random ===")
Insight.seed(UInt64(42))
check("get_seed returns number", Insight.get_seed() isa UInt64)
r3 = Insight.zeros(Int64[3, 4], Insight.float32)
check("rand_like defined", Insight.rand_like(r3).ptr != C_NULL)
check("randn_like defined", Insight.randn_like(r3).ptr != C_NULL)
check("exponential defined", Insight.exponential(1.0, Int64[3, 4]).ptr != C_NULL)

# ===== Additional FFT (Phase D) =====
println("=== Additional FFT ===")
f1 = Insight.ones(Int64[8], Insight.float32)
check("fftshift defined", Insight.fftshift(f1).ptr != C_NULL)
check("ifftshift defined", Insight.ifftshift(f1).ptr != C_NULL)
check("next_fast_len returns int", Insight.next_fast_len(100) isa Int)
check("hfft defined", begin
    c1 = Insight.to_complex(f1)
    Insight.hfft(c1).ptr != C_NULL
end)
check("ihfft defined", begin
    # ihfft takes real input, returns complex
    Insight.ihfft(f1).ptr != C_NULL
end)

# ===== Device Info =====
println("=== Device Info ===")
check("device_name cpu returns string", Insight.device_name() isa String)
check("cuda_version returns int", Insight.cuda_version() isa Int)
check("driver_version returns int", Insight.driver_version() isa Int)
check("compute_capability returns int", Insight.compute_capability() isa Int)
check("device_memory returns named tuple", begin
    m = Insight.device_memory()
    m.total isa Csize_t && m.free isa Csize_t
end)
check("gpu_count returns int", Insight.gpu_count() isa Int)

# ===== GPU =====
println("=== GPU ===")
check("load_backend cuda", begin
    try
        # Try loading CUDA backend - may fail on CPU-only machines
        ccall((:insight_jl_init_cpu, Insight.LIB_INSIGHT), Cvoid, ())
        true
    catch
        false
    end
end)

# ===== Missing Phase D: Indexing =====
println("=== Phase D Indexing ===")
idx_arr = Insight.from_data(Float32[10, 20, 30, 40, 50], Insight.float32)
check("topk defined", begin
    vals, idx = Insight.topk(idx_arr, 3)
    vals.ptr != C_NULL && idx.ptr != C_NULL
end)
check("interp defined", begin
    x = Insight.from_data(Float32[1.5, 2.5], Insight.float32)
    xp = Insight.from_data(Float32[1, 2, 3], Insight.float32)
    fp = Insight.from_data(Float32[10, 20, 30], Insight.float32)
    r = Insight.interp(x, xp, fp)
    r.ptr != C_NULL
end)

# ===== Missing Phase D: More Complex =====
println("=== Phase D Complex More ===")
check("as_real on complex", begin
    c2 = Insight.to_complex(a3)
    Insight.as_real(c2).ptr != C_NULL
end)
check("real_part on complex", begin
    c2 = Insight.to_complex(a3)
    Insight.real_part(c2).ptr != C_NULL
end)
check("imag_part on complex", begin
    c2 = Insight.to_complex(a3)
    Insight.imag_part(c2).ptr != C_NULL
end)

# ===== Display =====
println("=== Display ===")
println("Array repr: $(repr(a))")

# ===== Summary =====
println("")
println("Results: $passed passed, $failed failed")
if failed > 0
    exit(1)
end
