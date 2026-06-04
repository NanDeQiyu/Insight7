[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-1140%2B%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight

**轻量级、工业级张量计算框架，专为信号处理与 GPU 加速设计。**

Insight 是一个 C++ 张量库，设计理念深受 **PaddlePaddle**（算子注册机制、设备 HAL）、**Torch7**（简洁的 C API、TH/THC 架构精神）和 **NumPy/CuPy**（Python 侧 API 习惯）启发。提供 CPU/GPU 统一计算、零拷贝视图、动态算子派发和完整的信号处理支持。

## 特性

- **统一 API** -- `insight::Array` 无缝在 `CPUPlace` / `GPUPlace` 间切换
- **零拷贝视图** -- 通过 `strides` + `offset` 实现 `reshape`、`transpose`、`slice`
- **动态算子注册** -- `ops()["add"][CPU][F32](args)` 派发（Paddle 风格）
- **设备 HAL** -- 通过 `Device` 基类 + `extern "C"` 工厂实现 ABI 稳定的插件系统
- **信号处理** -- 89 个函数，覆盖 14 个子模块（窗函数、波形生成、B 样条、滤波器设计、卷积、滤波、频谱分析、小波、声学、雷达、解调、峰值检测、参数估计、I/O），全部配有 CPU 和 CUDA 后端 kernel
- **半精度支持** -- fp16/bf16 支持，通过 `half_utils.h` / `half_utils.cuh` 实现，116 个 kernel 文件包含半精度覆盖
- **语言绑定** -- Python（pybind11）、Lua（sol2）、Julia（ccall），按模块拆分的 wrapper，信号子命名空间
- **现代 C++** -- C++17/20，OpenMP 并行，FFTW3，OpenBLAS，cuBLAS，cuFFT
- **无自动求导** -- 保持库轻量、专注

## 架构

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API（elementwise, fft, signal, linalg 等）
│   ├── io/             # I/O（csv, print, sndfile）
│   ├── c_api/          # C ABI 接口（array, kernel, dtype, place）
│   └── plugin/         # 算子注册 + 设备 HAL
├── src/
│   ├── core/           # Array 实现、内存管理
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/kernels/    # CPU kernel（OpenMP + FFTW + OpenBLAS）
│   │   ├── cast/       ├── elementwise/   ├── fft/
│   │   ├── creation/   ├── indexing/      ├── linalg/
│   │   ├── manipulation/ ├── random/     ├── reduction/
│   │   ├── unary/      └── signal/       （14 个子目录）
│   └── cuda/kernels/   # CUDA kernel（cuBLAS + cuFFT + Thrust）
│       ├── cast/       ├── elementwise/   ├── fft/
│       ├── creation/   ├── indexing/      ├── linalg/
│       ├── manipulation/ ├── random/     ├── reduction/
│       ├── unary/      └── signal/       （14 个子目录）
├── bindings/
│   ├── python/insight/ # pybind11 绑定（按模块拆分的 wrapper）
│   ├── lua/insight/    # sol2 绑定（双调用约定）
│   └── julia/          # ccall 绑定（Insight.jl）
├── tests/
│   ├── cpu/            # CPU 测试（630+ 测试，27 个套件）
│   ├── cuda/           # CUDA 测试（510+ 测试，23 个套件）
│   └── python_align/   # NumPy 精度对齐测试（194 CPU + 190 CUDA）
└── demos/              # 示例程序（C++, Python, Lua, Julia）
```

## 快速开始

### 从源码编译

```bash
git clone https://github.com/PlumBlossomMaid/insight.git
cd insight
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON   # 启用 CUDA 后端
cmake --build . -j$(nproc)
```

### C++ 示例

```cpp
#include "insight/insight.h"
using namespace insight;

int main() {
    // 在 GPU 上创建张量
    Array a = ones({1000, 1000}, F32, GPUPlace(0));
    Array b = randn({1000, 1000}, F32, GPUPlace(0));

    // 矩阵乘法（自动选择 GPU kernel）
    Array c = matmul(a, b);

    // 移到 CPU 并访问数据
    Array cpu_c = c.to(CPUPlace());
    float value = cpu_c.at({0, 0}).item<float>();
}
```

### Python 示例

```python
import insight as ins

a = ins.zeros([2, 3], ins.float32)
b = ins.ones([2, 3], ins.float32)
c = a + b
s = ins.sum(c, axis=0)

# 信号处理
w = ins.signal.hann(256)
f, Pxx = ins.signal.welch(x, fs=1000)
```

### Lua 示例

```lua
local ins = require("insight")
ins.init({"cpu"})
local a = ins.zeros({2, 3}, ins.float32)
local b = ins.ones({2, 3}, ins.float32)
local c = a + b

-- 双调用约定
local w = ins.signal.hann(256)
local w2 = ins.signal.hann{n=256}
```

### Julia 示例

```julia
push!(LOAD_PATH, "/path/to/bindings/julia")
using Insight

a = Insight.zeros([2, 3], Insight.float32)
b = Insight.ones([2, 3], Insight.float32)
c = a + b
s = Insight.sum(c, axis=0)
```

## 依赖

| 依赖 | 版本 | 必需 | 说明 |
|------|------|------|------|
| CMake | 3.15+ | 是 | 构建系统 |
| C++17 编译器 | -- | 是 | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | 否 | 可选 GPU 后端 |
| OpenBLAS | 任意 | 否 | CPU 线性代数 |
| FFTW3 | 3.3+ | 否 | CPU FFT |
| OpenMP | -- | 否 | CPU 多线程 |
| Thrust | 捆绑 | 否 | CUDA 排序/去重 |
| GoogleTest | 自动 | -- | 自动获取 |

## 测试状态

**1140+ 测试全部通过** -- CPU（630+, 27 个套件）和 CUDA（510+, 23 个套件），另有 384 个精度对齐测试

| 套件 | CPU | CUDA | 备注 |
|------|-----|------|------|
| cast | 9 | 9 | |
| complex | 22 | 22 | |
| creation | 27 | 27 | |
| csv | 1 | 1 | |
| dtype | 9 | 9 | 共享 |
| elementwise | 28 | 28 | |
| fft | 19 | 19 | |
| indexing | 41 | 33 | |
| linalg | 43 | 43 | 15 原生 CUDA + 13 C_FALLBACK |
| manipulation | 42 | 42 | |
| operator | 50 | 50 | |
| print | 11 | 11 | |
| random | 31 | 31 | |
| reduction | 24 | 24 | |
| signal (核心) | 10 | 10 | 组合算子 |
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
| **合计** | **630+** | **510+** | |

### 精度对齐（对比 NumPy）

| 套件 | CPU | CUDA |
|------|-----|------|
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
| **合计** | **194** | **190** |

### 语言绑定测试

| 语言 | 测试框架 | 测试数 |
|------|---------|--------|
| Python | pytest | 76 基础 + 54 数值 + 384 对齐 |
| Lua | busted | 208 |
| Julia | Test stdlib | 74 |

## 示例程序

`demos/` 目录下提供 4 种语言、6 个场景的示例程序：

| 示例 | C++ | Python | Lua | Julia |
|------|-----|--------|-----|-------|
| basic_ops | 有 | 有 | 有 | 有 |
| fft_demo | 有 | 有 | 有 | 有 |
| gpu_transfer | 有 | 有 | 有 | 有 |
| linalg_demo | 有 | 有 | 有 | 有 |
| radar_task1 | 有 | 有 | 有 | 有 |
| sndfile_demo | 有 | 有 | 有 | 有 |

## 免责声明

> [!IMPORTANT]
> 本项目代码主要由 **DeepSeek**（深度求索）生成，是 **7 天快速原型开发** 的产物。作者完成了架构整合、测试验证和系统集成。
>
> 尽管所有 1140+ 个测试在 CPU 和 CUDA 上均稳定通过且完全对齐，但这仍然是一个**概念验证**项目。性能优化仍在进行中，在特定硬件配置或边界条件下可能存在未知 Bug。
>
> 本项目深度借鉴了 **PaddlePaddle** 的设计哲学（算子注册、设备 HAL），同时受 Torch7 和 NumPy/CuPy 启发。

## 许可证

[Apache License 2.0](LICENSE)

版权所有 2026 PlumBlossomMaid

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：
1. 代码遵循 `.clang-format` 风格
2. 所有现有测试通过
3. 新功能包含对应测试
