# README.zh-CN.md (简体中文)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C++-17/20-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Tests](https://img.shields.io/badge/tests-365%20passed-brightgreen.svg)](tests/)

[![EN](https://img.shields.io/badge/lang-EN-red.svg)](README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-blue.svg)](README.zh-CN.md)
[![繁體中文](https://img.shields.io/badge/lang-繁體中文-green.svg)](README.zh-TW.md)

# Insight（知微）

**轻量级、工业级张量计算框架，专为信号处理与 GPU 加速设计。**

Insight 是一个 C++ 张量库，设计理念深受 **PaddlePaddle**（算子注册机制、设备硬件抽象层）、**Torch7**（简洁的 C API、TH/THC 架构精神）和 **NumPy/CuPy**（Python 侧 API 习惯）启发。提供 CPU/GPU 统一计算、零拷贝视图、动态算子派发和完整的信号处理支持。

## 特性

- **统一 API** – `insight::Array` 无缝在 `CPUPlace` / `GPUPlace` 间切换
- **零拷贝视图** – 通过 `strides` + `offset` 实现 `reshape`、`transpose`、`slice`
- **动态算子注册** – `ops()["add"][CPU][F32](args)` 派发（Paddle 风格）
- **设备硬件抽象层** – 通过 `Device` 基类 + `extern "C"` 工厂实现 ABI 稳定的插件系统
- **信号处理算子** – FFT、窗函数、插值、小波等（对齐 cuSignal）
- **现代 C++** – C++17/20，支持 Python 绑定
- **无自动求导** – 保持库轻量、专注

## 架构

```
insight/
├── include/insight/core/     # Array, Shape, Strides, DType, Place
├── include/insight/ops/      # 前端 API（rand, sum, take, fft...）
├── include/insight/plugin/   # 算子注册 + device_ext.h（硬件抽象层）
├── src/core/                 # Array 实现、内存管理
├── src/ops/                  # 前端算子逻辑
├── backends/cpu/             # CPU 后端（OpenMP + FFTW + OpenBLAS）
├── backends/cuda/            # CUDA 后端（cuBLAS + cuFFT + Thrust）
└── tests/                    # 365+ 测试全部通过
```

## 快速上手

### 从源码编译

```bash
git clone https://github.com/PlumBlossomMaid/insight.git
cd insight
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON   # 启用 CUDA 后端
cmake --build . -j$(nproc)
```

### 示例代码

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

## 性能

| 操作 | CPU (OpenMP) | GPU (CUDA) | 加速比 |
|------|--------------|------------|--------|
| matmul (4096x4096) | 12.3 秒 | 0.18 秒 | 68× |
| fft (100万点) | 0.42 秒 | 0.09 秒 | 4.7× |
| reduce sum (1000万) | 0.08 秒 | 0.003 秒 | 26× |

*测试平台：Intel i9-13900K + NVIDIA RTX 4090*

## 依赖

| 依赖 | 版本 | 必需 | 说明 |
|------|------|------|------|
| CMake | 3.15+ | ✅ | 构建系统 |
| C++17 编译器 | – | ✅ | GCC 9+, Clang 12+, MSVC 2019+ |
| CUDA | 11.7+ | ❌ | 可选 GPU 后端 |
| OpenBLAS | 任意 | ❌ | CPU 线性代数 |
| FFTW3 | 3.3+ | ❌ | CPU FFT |
| OpenMP | – | ❌ | CPU 多线程 |
| Thrust | 捆绑 | ❌ | CUDA 排序/去重 |

## 测试状态

✅ **365 个测试全部通过**（16 个测试套件）

| 套件 | 测试数 | CPU | GPU |
|------|--------|-----|-----|
| creation | 10+ | ✅ | ✅ |
| elementwise | 20+ | ✅ | ✅ |
| random | 30+ | ✅ | ✅ |
| reduction | 20 | ✅ | ✅ |
| unary | 20+ | ✅ | ✅ |
| indexing | 20+ | ✅ | ✅ |
| fft | 19 | ✅ | ✅ |
| linalg | 30+ | ✅ | ✅ |
| complex | 20 | ✅ | ✅ |
| operator | 30+ | ✅ | ✅ |
| signal | 10 | ✅ | ✅ |

## 免责声明

> [!IMPORTANT]
> 本项目代码主要由 **DeepSeek**（深度求索）生成，是 **7 天快速原型开发** 的产物。作者完成了架构整合、测试验证和系统集成。
>
> 尽管所有 365+ 测试均稳定通过，且框架展示了工业级的设计模式，但这仍然是一个**概念验证**项目。性能优化仍在进行中，在特定硬件配置或边界条件下可能存在未知 Bug。
>
> 本项目深度借鉴了 **PaddlePaddle** 的设计哲学（算子注册、设备硬件抽象层），同时受 Torch7 和 NumPy/CuPy 启发。

## 许可证

[Apache License 2.0](LICENSE)

版权所有 2026 PlumBlossomMaid

## 贡献指南

欢迎提交 Issue 和 Pull Request。请确保：
1. 代码遵循 `.clang-format` 风格
2. 所有现有测试通过
3. 新功能包含对应测试