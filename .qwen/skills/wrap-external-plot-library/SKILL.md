---
name: wrap-external-plot-library
description: Wrapping an external C++ library (matplotplusplus) with Array conversion, conditional compilation, and separate CI
source: auto-skill
extracted_at: '2026-06-03T15:00:00.000Z'
---

# Wrap External Plot Library (matplotplusplus)

## Problem

Integrating a large external C++ library (matplotplusplus) into Insight7 while:
1. Converting `ins::Array` to the library's expected types (`std::vector<double>`)
2. Making the dependency optional (build works without it)
3. Keeping plot tests separate from core CPU/CUDA tests

## Solution

### 1. Conditional compilation with `#ifdef`

All plot code guarded by `INSIGHT_USE_MATPLOT`:

```cpp
// insight/ops/plot.h
#pragma once
#ifdef INSIGHT_USE_MATPLOT
// ... all declarations ...
#endif

// src/ops/plot.cpp
#ifdef INSIGHT_USE_MATPLOT
#include <matplot/matplot.h>
// ... all implementations ...
#endif

// tests/cpu/test_plot.cpp
#ifdef INSIGHT_USE_MATPLOT
// ... all tests ...
#endif
```

**Why:** The library may not be available (no gnuplot, no network). The build must work without it.

### 2. Array→vector conversion helpers

```cpp
vector_1d to_vector(const Array &a);  // handles F64/F32/I64/I32
vector_2d to_matrix(const Array &a);  // 2D only, row-major
```

Always transfer to CPU first: `if (a.place().kind() != DeviceKind::CPU) a = a.to(CPUPlace());`

### 3. Freestanding vs axes method

matplotplusplus has two calling styles:
- **Freestanding:** `mp::plot(x, y)` — operates on current axes via `gca()`
- **Axes method:** `mp::gca()->plot(x, y)` — explicit axes handle

**Rule:** Use freestanding by default. Switch to axes method when freestanding
template forwarding fails (common with `std::function` or `std::string` args).

```cpp
// Freestanding (works for most functions)
void plot(const Array &y, const std::string &fmt) {
    auto v = to_vector(y);
    fmt.empty() ? (void)mp::plot(v) : (void)mp::plot(v, fmt);
}

// Axes method (when freestanding fails)
void plot3(const Array &x, const Array &y, const Array &z,
           const std::string &fmt) {
    auto ax = mp::gca();
    fmt.empty() ? ax->plot3(to_vector(x), to_vector(y), to_vector(z))
                : ax->plot3(to_vector(x), to_vector(y), to_vector(z), fmt);
}

// No-op stub (when library has template bugs)
void ribbon(const Array &) {
    // matplotplusplus ribbon template forwarding issue
}
```

### 4. Separate CI workflow

```yaml
# .github/workflows/plot_tests.yml
name: Plot Module Tests
on:
  push:
    paths: ['include/insight/ops/plot.h', 'src/ops/plot.cpp', 'tests/cpu/test_plot.cpp']
  pull_request:
    paths: [...]

jobs:
  plot-tests:
    runs-on: ubuntu-latest
    steps:
    - name: Install dependencies
      run: sudo apt-get install -y gnuplot-nox  # headless rendering
    - name: Configure
      run: cmake .. -DINSIGHT_USE_MATPLOT=ON
    - name: Run plot tests
      run: ./tests/insight_tests_cpu --gtest_filter="PlotTestCPU.*"
    - name: Run full CPU tests (no regression)
      run: ./tests/insight_tests_cpu
```

**Why separate CI:** Plot tests need gnuplot, which is an extra system dependency.
Mixing them with core CPU/CUDA tests would add a hard dependency on gnuplot.

### 5. Test strategy

Without gnuplot installed, rendering tests crash. Test only what's safe:
- **to_vector / to_matrix conversions** — all dtype variants
- **Colormap enum** — verify distinct values
- **Link verification** — call all functions to verify they compile and link

```cpp
TEST_F(PlotTestCPU, ToVectorF64) {
    auto v = plot::to_vector(to_array(std::vector<double>{1, 2, 3}));
    ASSERT_EQ(v.size(), 3);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
}
```

## Known Issues

### matplotplusplus v1.2.1 template bugs

Some freestanding functions have broken template forwarding:
- `fimplicit(string)` — only `fimplicit(function)` works
- `fcontour(string)` — only `fcontour(function)` works
- `line(x, y)` — template deduction fails
- `binscatter(x, y)` — template deduction fails
- `ribbon(y)` — template deduction fails
- `pareto(x, y)` — template deduction fails

**Fix:** Use axes methods where possible, stub as no-op otherwise.
The `apply-third-party-patch` skill may be needed for deeper fixes.

### -fPIC for Python/Lua shared binding linking

When linking `insight_core` (which includes matplotplusplus as a static library)
into a Python/Lua `.so` (shared object), the static library must be compiled
with `-fPIC`. Otherwise the linker fails with:

```
relocation R_X86_64_PC32 against symbol `...' can not be used when making a shared object
```

**Fix:** Add `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` before the
`add_subdirectory` for matplotplusplus in the root CMakeLists.txt:

```cmake
if(INSIGHT_USE_MATPLOT)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # ← Required for Python/Lua .so
    add_subdirectory("${LOCAL_MATPLOT}" ...)
endif()
```

After changing this, must delete and rebuild the matplot library:
```bash
rm -f build/matplotplusplus/source/matplot/libmatplot.a
make -j$(nproc) insight_python
```

## Verification

```bash
# Build with MATPLOT=ON
cmake .. -DINSIGHT_USE_MATPLOT=ON && make -j$(nproc)

# Run plot tests
./tests/insight_tests_cpu --gtest_filter="PlotTestCPU.*"

# Build without MATPLOT (should still work)
cmake .. -DINSIGHT_USE_MATPLOT=OFF && make -j$(nproc)
./tests/insight_tests_cpu  # All tests pass, plot tests skipped
```

## gnuplot 5.4.2 `unknown` terminal font crash

gnuplot 5.4.2 (Ubuntu 22.04) doesn't support `font` option on the `unknown`
terminal. matplotplusplus sends `set terminal unknown font "Helvetica,10"`
which causes gnuplot to error, leading to a broken pipe and segfault.

**Fix:** Add `"unknown"` to the font blacklist in matplotplusplus:
`third_party/matplotplusplus/source/matplot/backend/gnuplot.cpp` line ~455:

```cpp
SV_CONSTEXPR std::string_view blacklist[] = {
    "dxf", "eepic", ..., "pdf", "unknown"};  // ← add "unknown"
```

**Why:** The `unknown` terminal is used internally by matplotplusplus for
bounding box computation. gnuplot 5.4.2's `unknown` terminal doesn't accept
`font`, but 5.4.4+ does. The blacklist prevents font option being sent.

**CI fix:** Use `ubuntu-24.04` (gnuplot 6.0) instead of `ubuntu-latest` (22.04).

## SIGPIPE handling for plot functions

matplotplusplus spawns gnuplot as a subprocess. If gnuplot crashes or the pipe
breaks, SIGPIPE kills the process. This is NOT a C++ exception — try/catch
won't help.

**Fix:** Add `signal(SIGPIPE, SIG_IGN)` before any plot calls:

```cpp
// C++ demo main()
#include <csignal>
#ifdef SIGPIPE
std::signal(SIGPIPE, SIG_IGN);
#endif

// Lua/Python bindings — inside #ifdef INSIGHT_USE_MATPLOT block
#include <csignal>
std::signal(SIGPIPE, SIG_IGN);
```

**Why:** SIGPIPE is a signal, not an exception. The default handler terminates
the process. Ignoring it causes the write to fail with EPIPE instead, which
matplotplusplus handles internally.

## C++ radar demo plot crash pattern

The radar demo generates detection results THEN plots. If plotting crashes,
all the computation is lost. Wrap plot calls in try/catch:

```cpp
try {
    save_plots(cpu_result, "task1_cpu");
} catch (const std::exception &e) {
    printf("[Warning] Plotting failed: %s\n", e.what());
}
```
