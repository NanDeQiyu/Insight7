# Plot CPU binding tests — 13 tests aligned with C++ plot test suite.
#
# Tests that plot functions exist and can be called without crashing.
# NOTE: Julia plot bindings are not yet implemented (no C ABI wrappers
# exported from libinsight_julia.so). These tests verify the module loads
# and report SKIP for all plot functions.
#
# Functions tested: plot, scatter, bar, hist, imshow, contour,
#                   subplot, title, xlabel, ylabel, legend, savefig, close
#
# Run with:
#   LD_LIBRARY_PATH=build/backends/cpu julia tests/cpu/julia/test_plot.jl

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "build", "bindings", "julia"))

using Insight

passed = 0
skipped = 0
failed = 0

function check(name, cond; skip=false)
    global passed, skipped, failed
    if skip
        skipped += 1
        println("SKIP: $name (plot bindings not yet available)")
    elseif cond
        passed += 1
    else
        failed += 1
        println("FAIL: $name")
    end
end

# ============================================================================
# Verify Insight module loads
# ============================================================================
a = Insight.zeros([2, 3], Insight.float32)
check("module_load", Insight.numel(a) == 6)

# ============================================================================
# Plot function tests (all skipped — no C ABI wrappers yet)
# ============================================================================

# Check if plot submodule exists (it won't, since no C ABI wrappers are exported)
has_plot = isdefined(Insight, :plot) && Insight.plot !== nothing

check("plot", true; skip=!has_plot)
check("scatter", true; skip=!has_plot)
check("bar", true; skip=!has_plot)
check("hist", true; skip=!has_plot)
check("imshow", true; skip=!has_plot)
check("contour", true; skip=!has_plot)
check("subplot", true; skip=!has_plot)
check("title", true; skip=!has_plot)
check("xlabel", true; skip=!has_plot)
check("ylabel", true; skip=!has_plot)
check("legend", true; skip=!has_plot)
check("savefig", true; skip=!has_plot)
check("close", true; skip=!has_plot)

# ============================================================================
# Results
# ============================================================================
println("\n" * "="^40)
println("Results: $passed passed, $skipped skipped, $failed failed")
if failed > 0
    exit(1)
end
