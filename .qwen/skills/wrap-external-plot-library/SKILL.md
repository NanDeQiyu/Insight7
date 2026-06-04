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

## gnuplot `unknown` terminal font crash (COMPREHENSIVE FIX)

**Root cause**: matplotplusplus's `terminal_has_font_option()` blacklist did NOT
include `"unknown"`. When CI runs headless (no display), gnuplot defaults to the
`unknown` terminal. The code sent `set terminal unknown font "Helvetica,10"`,
gnuplot rejected it, and pipe closure caused SIGPIPE → segfault.

This was the #1 recurring CI failure across 6 rounds of CI fixes.

**Fix (in `third_party/matplotplusplus/source/matplot/backend/gnuplot.cpp`)**:

1. Add `"unknown"` to font blacklist (~line 454):
```cpp
SV_CONSTEXPR std::string_view blacklist[] = {
    "dxf", "eepic", ..., "pdf", "unknown"};  // ← add "unknown"
```

2. Fallback from "unknown" to "pngcairo" terminal (supports full features):
```cpp
// pngcairo supports pm3d/image rendering (required for imshow/contour).
// "png" (libgd) lacks pm3d support — crashes on imshow.
if (terminal_ == "unknown") {
    if (terminal_is_available("pngcairo")) {
        terminal_ = "pngcairo";
    } else if (terminal_is_available("png")) {
        terminal_ = "png";
    } else {
        terminal_ = "dumb";
    }
}
```

**Why pngcairo first**: "png" uses libgd which does NOT support `set pm3d map`
(required by imshow/contour). Using "png" causes SIGSEGV on image rendering.

3. **Redirect gnuplot stdout to /dev/null (CRITICAL — fixes binary output leak)**:
```cpp
// gnuplot.cpp constructor:
pipe_ = POPEN("gnuplot >/dev/null 2>&1", "w");
```
This prevents gnuplot from writing binary PNG data to the parent's stdout.
When `save()` sends `set output "filename"`, gnuplot writes to the file.
Any other output goes to /dev/null. **All languages benefit from this.**

4. Check gnuplot binary exists before opening pipe:
```cpp
if (system("gnuplot --version > /dev/null 2>&1") != 0) {
    pipe_ = nullptr;
    return;
}
```

5. Graceful pipe operations with ferror/EOF checks:
```cpp
// In flush_commands():
if (!pipe_) { return false; }
if (ferror(pipe_)) { PCLOSE(pipe_); pipe_ = nullptr; return false; }
if (fputs("\n", pipe_) == EOF || fflush(pipe_) == EOF) {
    PCLOSE(pipe_); pipe_ = nullptr; return false;
}

// In run_command():
if (!pipe_) { return; }
if (ferror(pipe_)) { PCLOSE(pipe_); pipe_ = nullptr; return; }
if (fputs(command.c_str(), pipe_) == EOF) {
    PCLOSE(pipe_); pipe_ = nullptr; return;
}

// In destructor:
if (!pipe_) return;
errno = 0;
if (fputs("set output\nexit\n", pipe_) != EOF) { fflush(pipe_); }
PCLOSE(pipe_);
pipe_ = nullptr;
```

**Why pngcairo, not dumb**: pngcairo supports `save()` to PNG files (essential for
batch plotting). "dumb" only outputs text to console, which is useless for the
competition's 1000-frame batch processing workflow.

**Patch management**: Save as `patches/matplotplusplus/gnuplot-unknown-terminal.patch`
and register in CMakeLists.txt for both local and FetchContent builds.

## Python plot tests: imshow/contour run directly (no subprocess)

**Problem**: `ins.plot.imshow()` and `ins.plot.contour()` crashed in headless CI.

**Root cause**: The gnuplot terminal fallback checked `pngcairo` first. pngcairo
depends on cairo/pango which SIGSEGV in headless environments on image rendering.

**Fix**: Swap the terminal fallback order — prefer `png` (libgd) over `pngcairo`
(cairo/pango). libgd works without a display and supports image rendering.

With this fix, imshow/contour run directly in the test (no subprocess isolation needed):

```python
def test_imshow_basic(self):
    data = ins.from_numpy(np.random.rand(5, 5))
    self._save_first()
    ins.plot.imshow(data)
    ins.plot.clf()

def test_contour_basic(self):
    X = ins.from_numpy(xx)
    Y = ins.from_numpy(yy)
    Z = ins.from_numpy(zz)
    self._save_first()
    ins.plot.contour(X, Y, Z)
    ins.plot.clf()
```

**Why NOT subprocess isolation**: Subprocess (fork) inherits the parent's gnuplot
pipe state, causing pipe corruption. Even with `spawn`, subprocess isolation hides
crashes — the test shows "PASSED" while the subprocess segfaults. Direct execution
is honest: if it crashes, the test fails.

**CI workflow**: No special handling needed. Plot tests pass directly.

## Plot to PNG files (not console)

**Requirement**: Competition needs batch processing 1000 frames → save each frame
as PNG file for later analysis. Plotting must save to files, not dump to console.

**Pattern**: Call `save()` AFTER `figure()` but BEFORE any plot commands:

```cpp
// C++ — save() after figure() ensures output goes to file
plot::figure();
plot::save("output.png");  // sets gnuplot's "set output"
plot::imshow(data);
plot::colorbar();
plot::title("Range-Doppler Map");
```

```python
# Python — same pattern
ins.plot.figure()
ins.plot.save("output.png")
ins.plot.imshow(data)
ins.plot.title("Range-Doppler Map")
```

**C++ fork isolation for demos**: Since gnuplot pipe cleanup can segfault even
with pngcairo terminal, run plot operations in a child process:

```cpp
static void save_plots(const Result &result, const char *prefix) {
    // Pre-compute data in parent
    Array energy_arr = to_array(result.energy, ...);

    // Fork: child does plotting, parent waits
    pid_t pid = fork();
    if (pid == 0) {
        plot::figure(); plot::save(std::string(prefix) + ".png");
        plot::imshow(energy_arr);
        _exit(0);  // child exits after plotting
    } else if (pid > 0) {
        int s; waitpid(pid, &s, 0);
        printf("已保存: %s.png\n", prefix);
    }
}
```

**Why fork()**: The gnuplot pipe cleanup in the destructor can segfault.
`SIG_IGN` for SIGSEGV is unreliable on Linux. `fork()` is the only reliable
isolation mechanism — the child's crash doesn't affect the parent.

## C++ demo GPU crash isolation

**Problem**: `ins::init({"cuda"})` may succeed (loads library) but `set_device(GPUPlace(0))`
fails because no physical GPU exists. This causes an unhandled exception → crash.

**Fix**: Wrap the entire GPU section in try-catch:

```cpp
if (has_gpu) {
    try {
        set_device(GPUPlace(0));
        auto gpu = run_task1(GPUPlace(0));
        // ... print results ...
    } catch (const std::exception &e) {
        printf("[提示] GPU 不可用: %s\n", e.what());
    } catch (...) {
        printf("[提示] GPU 不可用\n");
    }
}
```

**Why not check `has_device()`**: The `init({"cuda"})` call may succeed even
without a physical GPU (it just loads the shared library). The actual GPU
availability is only known when trying to use it.

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
```

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
