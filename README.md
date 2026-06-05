[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**A lightweight, industrial-grade tensor computation framework for signal processing and GPU acceleration.**

Insight is a C++ tensor library inspired by **PaddlePaddle** (operator registration, device HAL), **Torch7** (clean C API, TH/THC spirit), and **NumPy/CuPy** (Python-side API conventions). It delivers CPU/GPU unified computing with zero-copy views, dynamic operator dispatch, and full signal processing support.

## Features

- **Unified API** -- `insight::Array` works seamlessly on `CPUPlace` and `GPUPlace`
- **Zero-Copy Views** -- `reshape`, `transpose`, `slice` via strides & offset
- **Dynamic Operator Registry** -- `ops()["add"][CPU][F32](args)` dispatch (Paddle style)
- **Device HAL** -- ABI-stable plugin system via `Device` base class + `extern "C"` factory
- **Signal Processing** -- 89 functions across 14 submodules (windows, waveforms, B-splines, filter design, convolution, filtering, spectral analysis, wavelets, acoustics, radar, demod, peak finding, estimation, I/O), all with CPU and CUDA backend kernels
- **Half-Precision** -- fp16/bf16 support via `half_utils.h`/`half_utils.cuh`, 116 kernel files with half-precision coverage
- **Language Bindings** -- Python (pybind11), Lua (sol2), Julia (ccall) with per-module wrappers and signal sub-namespaces
- **Modern C++** -- C++17/20, OpenMP parallel, FFTW3, OpenBLAS, cuBLAS, cuFFT
- **No Autograd** -- Keeps the library lightweight and focused

## Architecture

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # Frontend API (elementwise, fft, signal, linalg, etc.)
│   ├── io/             # I/O (csv, print, sndfile)
│   ├── c_api/          # C ABI interfaces (array, kernel, dtype, place)
│   └── plugin/         # Operator registry + device HAL
├── src/
│   ├── core/           # Array implementation, memory management
│   ├── ops/            # Frontend operator logic
│   └── internal/       # Internal utilities
├── backends/
│   ├── cpu/kernels/    # CPU kernels (OpenMP + FFTW + OpenBLAS)
│   │   ├── cast/       ├── elementwise/   ├── fft/
│   │   ├── creation/   ├── indexing/      ├── linalg/
│   │   ├── manipulation/ ├── random/     ├── reduction/
│   │   ├── unary/      └── signal/       (14 subdirectories)
│   └── cuda/kernels/   # CUDA kernels (cuBLAS + cuFFT + Thrust)
│       ├── cast/       ├── elementwise/   ├── fft/
│       ├── creation/   ├── indexing/      ├── linalg/
│       ├── manipulation/ ├── random/     ├── reduction/
│       ├── unary/      └── signal/       (14 subdirectories)
├── bindings/
│   ├── python/insight/ # pybind11 bindings (per-module wrappers)
│   ├── lua/insight/    # sol2 bindings (dual calling convention)
│   └── julia/          # ccall bindings (Insight.jl)
├── tests/
│   ├── cpu/            # CPU tests (630+ tests, 27 suites)
│   ├── cuda/           # CUDA tests (510+ tests, 23 suites)
│   └── python_align/   # NumPy precision alignment tests (194 CPU + 190 CUDA)
└── demos/              # Example programs (C++, Python, Lua, Julia)
```

## Quick Start

### Build from Source

```bash
git clone https://github.com/PlumBlossomMaid/insight.git
cd insight
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON   # enable CUDA backend
cmake --build . -j$(nproc)
```

### C++ Example

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // Create arrays on GPU
    Array a = ones({1000, 1000}, F32, GPUPlace(0));
    Array b = randn({1000, 1000}, F32, GPUPlace(0));

    // Matrix multiplication (automatically picks GPU kernel)
    Array c = matmul(a, b);

    // Move to CPU and access data
    Array cpu_c = c.to(CPUPlace());
    float value = cpu_c.at({0, 0}).item<float>();
}
```

### Python Example

```python
import insight as ins

a = ins.zeros([2, 3], ins.float32)
b = ins.ones([2, 3], ins.float32)
c = a + b
s = ins.sum(c, axis=0)

# NumPy-style indexing
row = a[1]          # partial indexing → shape (3,)
val = a[1, 2]       # scalar extraction
sub = a[1:, ::2]    # mixed slice indexing

# Signal processing
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua Example

```lua
local ins = require("insight")
-- Backend auto-detected, no manual init needed

local a = ins.zeros({2, 3}, ins.float32)
local b = ins.ones({2, 3}, ins.float32)
local c = a + b
local s = ins.sum(c)

-- NumPy-style indexing (1-based for Lua)
local row = a[1]        -- partial indexing → shape (3,)
local val = a[1][2]     -- scalar extraction

-- Dual calling convention
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}

-- Device management
print(ins.get_device())  -- cuda:0 or cpu:0
```

### Julia Example

```julia
push!(LOAD_PATH, "/path/to/bindings/julia")
using Insight

a = Insight.zeros(Int64[2, 3], Insight.float32)
b = Insight.ones(Int64[2, 3], Insight.float32)
c = a + b
s = Insight.sum(c)

# NumPy-style indexing (1-based for Julia)
row = a[1]          # partial indexing → shape (3,)
val = a[1, 2]       # scalar extraction

# Device management
dt, id = Insight.get_device()  # (0, 0) for CPU, (1, 0) for GPU
```

## Install Language Bindings

### Python

```bash
pip install -e .   # editable install from project root
```

### Lua (via luarocks)

```bash
# Lua 5.3
luarocks make bindings/lua/insight-1.0-1.rockspec LUA_DIR=/usr --local

# LuaJIT
luarocks make bindings/lua/insight-1.0-1.rockspec --local
```

### Julia

```julia
push!(LOAD_PATH, "/path/to/Insight7/bindings/julia")
using Insight
```

## Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| CMake | 3.15+ | Yes | Build system |
| C++17 compiler | -- | Yes | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | No | Optional GPU backend |
| OpenBLAS | any | No | CPU linear algebra |
| FFTW3 | 3.3+ | No | CPU FFT |
| OpenMP | -- | No | CPU multi-threading |
| Thrust | bundled | No | CUDA sorting/unique |
| GoogleTest | auto | -- | Automatically fetched |

## Test Status

**1140+ tests passing** -- CPU (630+, 27 suites) and CUDA (510+, 23 suites), plus 384 precision alignment tests

| Suite | CPU | CUDA | Notes |
|-------|-----|------|-------|
| cast | 9 | 9 | |
| complex | 22 | 22 | |
| creation | 27 | 27 | |
| csv | 1 | 1 | |
| dtype | 9 | 9 | Shared |
| elementwise | 28 | 28 | |
| fft | 19 | 19 | |
| indexing | 41 | 33 | |
| linalg | 43 | 43 | 15 native CUDA + 13 C_FALLBACK |
| manipulation | 42 | 42 | |
| operator | 50 | 50 | |
| print | 11 | 11 | |
| random | 31 | 31 | |
| reduction | 24 | 24 | |
| signal (core) | 10 | 10 | Composite ops |
| signal_windows | 30 | 30 | |
| signal_waveforms | 18 | 18 | |
| signal_bsplines | 13 | 13 | |
| signal_filter_design | 22 | 22 | |
| signal_convolution | 21 | 17 | |
| signal_filtering | 23 | 15 | |
| signal_spectral | 11 | -- | |
| signal_wavelets | 13 | -- | |
| signal_acoustics | 9 | -- | |
| signal_radar | 7 | -- | |
| signal_io | 11 | -- | |
| signal_peak_finding | 3 | -- | |
| signal_demod | 1 | -- | |
| signal_estimation | 1 | -- | |
| plot | 13 | -- | |
| unary | 27 | 27 | |
| audio | 9 | 9 | |
| **Total** | **630+** | **510+** | |

### Precision Alignment (vs NumPy)

| Suite | CPU | CUDA |
|-------|-----|------|
| cast | 14 | 14 |
| complex | 8 | 8 |
| creation | 14 | 14 |
| elementwise | 24 | 24 |
| fft | 18 | 18 |
| linalg | 22 | 22 |
| manipulation | 18 | 18 |
| reduction | 22 | 22 |
| signal | 20 | 16 |
| unary | 24 | 24 |
| numerical | 10 | 10 |
| **Total** | **194** | **190** |

### Language Binding Tests

| Language | Test Framework | Tests |
|----------|---------------|-------|
| Python | pytest | 76 smoke + 54 numerical + 384 alignment |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## Demos

Example programs in `demos/` covering 4 languages and 6 scenarios:

| Demo | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | Yes | Yes | Yes | Yes |
| fft_demo | Yes | Yes | Yes | Yes |
| gpu_transfer | Yes | Yes | Yes | Yes |
| linalg_demo | Yes | Yes | Yes | Yes |
| radar_task1 | Yes | Yes | Yes | Yes |
| sndfile_demo | Yes | Yes | Yes | Yes |

## Disclaimer

> [!IMPORTANT]
> This codebase was primarily generated by **DeepSeek** (深度求索) as part of a 7-day rapid prototyping challenge. The author integrated, tested, and verified the core architecture.
>
> While all 1140+ tests pass reliably on both CPU and CUDA with fully aligned coverage, this is still a **proof-of-concept**. Performance optimizations are ongoing, and you may encounter bugs in edge cases or with specific hardware configurations.
>
> This project draws deeply from **PaddlePaddle's** design philosophy (operator registration, device HAL). Also inspired by Torch7 and NumPy/CuPy.

## License

[Apache License 2.0](LICENSE)

Copyright 2026 PlumBlossomMaid

## Contributing

Issues and pull requests are welcome. Please ensure:
1. Code follows `.clang-format` style
2. All existing tests pass
3. New features include corresponding tests
