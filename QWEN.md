# Insight7 - 科学计算框架

## 项目概述

Insight 是一个轻量级工业级 C++ 张量计算框架，用于信号处理和 GPU 加速。

**设计灵感**：
- **PaddlePaddle** — 算子注册机制、设备 HAL 抽象
- **Torch7** — 干净 C API、TH/THC 精神
- **NumPy/CuPy** — Python 端 API 约定

**当前状态（2026-06-07 PR #40-43 已合并）**：
- ✅ CPU 后端完整实现（650+ 测试全部通过）
- ✅ CUDA 后端完整实现（606+ 测试全部通过，与 CPU 完全对齐）
- ✅ C_FALLBACK 机制已修复，支持 GPU→CPU 自动数据传输
- ✅ signal 模块全部完成（89 函数，14 子模块，CPU/CUDA backend kernels 全部完成）
- ✅ fp16/bf16 支持：half_utils.h/cuh + 116 个 kernel 文件覆盖
- ✅ plot 模块已实现（ins::plot，70+ 函数，matplotplusplus 集成）
- ✅ audio 模块已实现（ins::audio，支持 WAV/FLAC/OGG 读写）
- ✅ Python 风格多维字符串切片（arr[":,1:-1"] 语法）
- ✅ NumPy 风格部分索引（a[1] on multi-dim 返回切片，4 语言受益）
- ✅ reshape({-1}) 推断维度 + dtype_of<long long> 特化（PR #41）
- ✅ 语言绑定：Python (pybind11) / Lua (sol2) / Julia (ccall)
  - Python: pip install .，@ 运算符，NumPy 索引
  - Lua: luarocks install，1-based 索引，双调用约定，ins.signal.cwt 绑定
  - Julia: ccall，getindex，get_device/set_device
- ✅ NumPy 数值正确性对齐（194 CPU + 190 CUDA precision tests）
- ✅ CI 全部通过（14 个 job，路径触发器已优化）
- ✅ 比赛 Demo 完成：
  - radar_detection（任务1）：4 语言（C++/Python/Lua/Julia），GPU ~4ms/帧
  - radar_pipeline（任务2）：4 语言，GPU C++/Python ~3.5s/帧
  - 两个 demo 均支持 --device cpu|gpu --seed --iterations --plot
- ✅ 代码规模：~95,000 行（C++/CUDA/Python/Lua/Julia/CMake）
- 🔧 待补全：7 个 cuSignal 附录算子（channelize_poly, sosfilt, upfirdn, firfilter2, qmf, istft, lombscargle 前端）
- 🎯 下一步：补全算子 + cudaEvent 计时 + CPU 验证程序 + Windows 适配

## 架构

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API 声明 (elementwise.h, fft.h, signal.h, etc.)
│   ├── io/             # I/O (csv.h, print.h, sndfile.h)
│   ├── c_api/          # C ABI 接口 (array.h, kernel.h, dtype.h, place.h)
│   └── plugin/         # 算子注册 + 设备 HAL
├── src/
│   ├── core/           # Array 实现、内存管理
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/kernels/    # CPU 内核实现 (OpenMP + FFTW + OpenBLAS)
│   │   ├── cast/ creation/ elementwise/ fft/ indexing/
│   │   ├── linalg/ manipulation/ random/ reduction/ unary/
│   │   └── signal/ (14 子目录: acoustics, bsplines, convolution, demod,
│   │       estimation, filter_design, filtering, io, peak_finding,
│   │       radar, spectral, waveforms, wavelets, windows)
│   └── cuda/kernels/   # CUDA 内核实现 (cuBLAS + cuFFT + Thrust)
│       ├── cast/ creation/ elementwise/ fft/ indexing/
│       ├── linalg/ manipulation/ random/ reduction/ unary/
│       └── signal/ (14 子目录，与 CPU 完全对齐)
├── bindings/
│   ├── python/insight/ # pybind11 绑定 (per-module wrappers + signal/ 子命名空间)
│   ├── lua/insight/    # sol2 绑定 (dual calling convention + _wrap.lua 共享工具)
│   └── julia/          # ccall 绑定 (Insight.jl + modules/signal/ 子命名空间)
├── tests/
│   ├── cpu/            # CPU 测试 (630+ tests, 27 suites)
│   ├── cuda/           # CUDA 测试 (510+ tests, 23 suites, 完全对齐)
│   └── python_align/   # NumPy 精度对齐测试 (194 CPU + 190 CUDA)
└── demos/              # 示例程序 (C++/Python/Lua/Julia, 6 demos)
```

## 构建系统

### 快速构建

```bash
mkdir build && cd build
cmake .. -DINSIGHT_WITH_CUDA=ON
cmake --build . -j$(nproc)
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `INSIGHT_WITH_CUDA` | ON | 启用 CUDA 后端 |
| `INSIGHT_BUILD_TESTS` | ON | 构建单元测试 |
| `INSIGHT_USE_OPENBLAS` | ON | 使用 OpenBLAS 加速线性代数 |
| `INSIGHT_USE_FFTW3` | ON | 使用 FFTW3 加速 FFT |
| `INSIGHT_DYNAMIC_CPU_BACKEND` | ON | CPU 后端动态链接 |
| `INSIGHT_DYNAMIC_GPU_BACKEND` | ON | GPU 后端动态链接 |

### 依赖

- **C++17 编译器**：GCC 9+, Clang 12+, MSVC 2019+
- **CMake 3.15+**
- **CUDA 11.7+**（可选）
- **OpenBLAS**（可选，线性代数）
- **FFTW3**（可选，FFT）
- **OpenMP**（可选，CPU 多线程）
- **GoogleTest**（自动获取）

## 代码编写规范

### 1. CPU Kernel 模板

**文件位置**：`backends/cpu/kernels/<module>/<op>.cpp`

```cpp
// backends/cpu/kernels/elementwise/add.cpp
#include "../../registry/cpu_registry.h"
#include "common.h"  // 模块级 common.h
#include "insight/c_api/array.h"

extern "C" {

C_Status add_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* a = (InsightArray*)inputs[0];
    InsightArray* b = (InsightArray*)inputs[1];
    InsightArray* out = (InsightArray*)outputs[0];

    if (!a || !b || !out) {
        cpu_set_last_error("add: null array pointer");
        return C_FAILED;
    }

    switch (out->dtype) {
        case INSIGHT_DTYPE_F32:
            ELEMENTWISE_KERNEL_LOOP(float, +);
            break;
        case INSIGHT_DTYPE_F64:
            ELEMENTWISE_KERNEL_LOOP(double, +);
            break;
        // ... 其他类型
        default:
            cpu_set_last_error("add: unsupported dtype");
            return C_FAILED;
    }

    return C_SUCCESS;
}

}  // extern "C"

REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_F32, add_kernel_cpu);
REGISTER_CPU_KERNEL(add, INSIGHT_DTYPE_F64, add_kernel_cpu);
// ... 注册所有支持的类型
```

### 2. CUDA Kernel 模板

**文件位置**：`backends/cuda/kernels/<module>/<op>.cu`

```cpp
// backends/cuda/kernels/elementwise/add.cu
#include "../../registry/cuda_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"

template<typename T>
__global__ void add_kernel(
    const T* a, const T* b, T* out,
    const ElementwiseMetadata* meta) {

    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= meta->numel) return;

    int64_t a_off = elementwise_offset(linear, meta, meta->a_strides);
    int64_t b_off = elementwise_offset(linear, meta, meta->b_strides);
    int64_t out_off = elementwise_offset(linear, meta, meta->out_strides);

    out[out_off] = a[a_off] + b[b_off];
}

extern "C" {

C_Status add_kernel_gpu(void** inputs, void** outputs) {
    InsightArray* a = (InsightArray*)inputs[0];
    InsightArray* b = (InsightArray*)inputs[1];
    InsightArray* out = (InsightArray*)outputs[0];

    if (!a || !b || !out) {
        gpu_set_last_error("add: null array pointer");
        return C_FAILED;
    }

    ElementwiseMetadata* meta = alloc_elementwise_metadata(a, b, out);
    dim3 blocks = elementwise_blocks(out->numel);
    dim3 threads = elementwise_threads();

    switch (out->dtype) {
        case INSIGHT_DTYPE_F32:
            add_kernel<float><<<blocks, threads>>>(
                (float*)a->data, (float*)b->data, (float*)out->data, meta);
            break;
        // ... 其他类型
        default:
            free_elementwise_metadata(meta);
            gpu_set_last_error("add: unsupported dtype");
            return C_FAILED;
    }

    cudaError_t err = cudaGetLastError();
    free_elementwise_metadata(meta);

    if (err != cudaSuccess) {
        gpu_set_last_error(cudaGetErrorString(err));
        return C_FAILED;
    }

    return C_SUCCESS;
}

}  // extern "C"

REGISTER_GPU_KERNEL(add, INSIGHT_DTYPE_F32, add_kernel_gpu);
// ... 注册所有支持的类型
```

### 3. 常用宏和工具

#### CPU 后端

| 宏 | 位置 | 用途 |
|----|------|------|
| `UNARY_KERNEL_LOOP(CTYPE, Func)` | `unary/common.h` | 一元操作，stride 感知，OpenMP 并行 |
| `UNARY_CMP_LOOP(CTYPE, Func)` | `unary/common.h` | 返回 bool 的一元操作 |
| `ELEMENTWISE_KERNEL_LOOP(CTYPE, OP)` | `elementwise/common.h` | 二元算术操作 |
| `ELEMENTWISE_CMP_LOOP(CTYPE, OP)` | `elementwise/common.h` | 二元比较操作 |
| `REGISTER_CPU_KERNEL(op, dtype, func)` | `cpu_registry.h` | 注册 CPU kernel |

#### GPU 后端

| 工具 | 位置 | 用途 |
|------|------|------|
| `elementwise_blocks(n)` | `elementwise/common.cuh` | 计算 grid 维度 (256 线程/block) |
| `elementwise_threads()` | `elementwise/common.cuh` | 返回线程维度 (256) |
| `ElementwiseMetadata` | `elementwise/common.cuh` | 设备端 stride/shape 信息 |
| `alloc_elementwise_metadata(a, b, out)` | `elementwise/common.cuh` | 分配并拷贝 metadata 到设备 |
| `free_elementwise_metadata(meta)` | `elementwise/common.cuh` | 释放设备 metadata |
| `elementwise_offset()` | `elementwise/common.cuh` | `__device__` 线性索引转偏移 |
| `REGISTER_GPU_KERNEL(op, dtype, func)` | `cuda_registry.h` | 注册 GPU kernel |

### 4. 数据类型映射

| Insight DType | C/C++ 类型 | CUDA 类型 |
|---------------|-----------|-----------|
| `INSIGHT_DTYPE_BOOL` | `bool` | `bool` |
| `INSIGHT_DTYPE_U8` | `uint8_t` | `uint8_t` |
| `INSIGHT_DTYPE_I8` | `int8_t` | `int8_t` |
| `INSIGHT_DTYPE_I16` | `int16_t` | `int16_t` |
| `INSIGHT_DTYPE_I32` | `int32_t` | `int32_t` |
| `INSIGHT_DTYPE_I64` | `int64_t` | `int64_t` |
| `INSIGHT_DTYPE_U16` | `uint16_t` | `uint16_t` |
| `INSIGHT_DTYPE_U32` | `uint32_t` | `uint32_t` |
| `INSIGHT_DTYPE_U64` | `uint64_t` | `uint64_t` |
| `INSIGHT_DTYPE_F32` | `float` | `float` |
| `INSIGHT_DTYPE_F64` | `double` | `double` |
| `INSIGHT_DTYPE_C32` | `std::complex<float>` | `cuFloatComplex` |
| `INSIGHT_DTYPE_C64` | `std::complex<double>` | `cuDoubleComplex` |

### 5. 错误处理规范

```cpp
// 设置错误信息
cpu_set_last_error("operation: specific error message");

// 获取错误信息（用于检查）
const char* err = cpu_get_last_error();
if (err && err[0] != '\0') {
    // 有错误
    return C_FAILED;
}
```

**规则**：
1. 每个 kernel 入口检查所有输入指针
2. 内存分配失败时设置错误并返回
3. LAPACK/CUDA 调用失败时设置错误并返回
4. 成功返回前清除错误信息：`cpu_set_last_error("")`

### 6. 注册类型范围

| 操作类别 | 注册类型 |
|----------|----------|
| 算术运算 (add, sub, mul, div) | 所有数值类型 + bool + 复数 |
| 比较运算 (equal, greater, less) | 所有数值类型 + bool + 复数 |
| 逻辑运算 (logical_and, or, xor) | 仅 bool |
| 位运算 (bitwise_and, or, xor) | 整数类型 + bool |
| 移位运算 | 整数类型 |
| 一元运算 (abs, sin, exp) | 浮点 + 复数（视函数而定）|
| 规约运算 | 浮点 + 整数 |

## CUDA 后端状态

### 全部模块已完成（CPU/CUDA 测试完全对齐）

| 模块 | CPU 测试 | CUDA 测试 | 实现方式 |
|------|---------|----------|---------|
| cast | 9 | 9 | 原生 CUDA |
| complex | 22 | 22 | 原生 CUDA |
| creation | 27 | 27 | 原生 CUDA |
| csv | 1 | 1 | 原生 CUDA |
| dtype | 9 | 9 | 共享 |
| elementwise | 28 | 28 | 原生 CUDA |
| fft | 19 | 19 | 原生 CUDA |
| indexing | 33 | 33 | 原生 CUDA |
| linalg | 43 | 43 | 15 原生 CUDA + 13 C_FALLBACK |
| manipulation | 42 | 42 | 原生 CUDA |
| operator | 50 | 50 | 原生 CUDA |
| print | 11 | 11 | 原生 CUDA |
| random | 31 | 31 | 原生 CUDA |
| reduction | 24 | 24 | 原生 CUDA |
| signal (core) | 10 | 10 | 组合算子（底层 primitive 已有 CUDA kernel） |
| signal_windows | 30 | 30 | 原生 CUDA backend kernels |
| signal_waveforms | 18 | 18 | 原生 CUDA backend kernels |
| signal_bsplines | 13 | 13 | 原生 CUDA backend kernels |
| signal_filter_design | 22 | 22 | 原生 CUDA backend kernels |
| signal_convolution | 21 | 17 | 原生 CUDA backend kernels |
| signal_filtering | 23 | 15 | 原生 CUDA backend kernels |
| signal_spectral | 11 | — | CPU backend kernels (CUDA 待补充) |
| signal_wavelets | 13 | — | CPU backend kernels (CUDA 待补充) |
| signal_acoustics | 9 | — | CPU backend kernels (CUDA 待补充) |
| signal_radar | 7 | — | CPU backend kernels (CUDA 待补充) |
| signal_io | 11 | — | CPU backend kernels (CUDA 待补充) |
| signal_peak_finding | 3 | — | CPU backend kernels (CUDA 待补充) |
| signal_demod | 1 | — | CPU backend kernels (CUDA 待补充) |
| signal_estimation | 1 | — | CPU backend kernels (CUDA 待补充) |
| unary | 27 | 27 | 原生 CUDA |
| audio | 9 | 9 | CPU 读写 + GPU 传输验证 |
| indexing (string slice) | 8 (+8) | — | 多维字符串切片测试 |
| **总计** | **630+** | **510+** | |

> **注意**：signal 子模块的 14 个 CPU backend kernels 目录全部完成（acoustics, bsplines, convolution, demod, estimation, filter_design, filtering, io, peak_finding, radar, spectral, waveforms, wavelets, windows）。CUDA backend kernels 的 14 个目录也全部完成，但部分子模块的 CUDA 测试尚待补充。

Linalg 原生 CUDA kernel（cuBLAS/cuSOLVER）：matmul, dot, outer, trace, matrix_power, det, slogdet, inv, solve, svd, svdvals, eigh, eigvalsh, solve_triangular, norm

C_FALLBACK（自动 GPU→CPU 数据传输）：cholesky, cholesky_solve, qr, lq, lu, lu_unpack, cond, matrix_rank, pinv, lstsq, eig, eigvals, cov

## 语言绑定

| 语言 | 绑定库 | 模块 | 测试 |
|------|--------|------|------|
| Python | pybind11 | `bindings/python/insight/` | 76 smoke + 54 numerical + 384 alignment |
| Lua | sol2 | `bindings/lua/insight/` | 208 (busted) |
| Julia | ccall | `bindings/julia/Insight.jl` | 74 |

API 风格（PaddlePaddle）：`ins.float32`, `ins.CPUPlace()`, `ins.zeros([2,3])`

**Per-module wrappers**：每种语言都有与 C++ 源码结构对齐的 per-module wrapper 文件，包括完整的 signal 子命名空间（14 个子模块）。

Lua 1-based 索引自动转换（`a[1]` → 首元素，`a["2:4"]` → C++ `[1:3]`）。

**Lua `_wrap.lua` 共享工具**：所有 Lua 函数通过 `insight/_wrap.lua` 实现 dual calling convention，支持 positional (`func(a, b)`) 和 named-table (`func{a=x, b=y}`) 两种调用方式。该工具自动检测调用方式，通过检查是否有非数字 key 来判断。

## 测试

```bash
# 运行所有测试
cd build && ctest

# 运行特定模块（CPU）
./tests/insight_tests_cpu --gtest_filter="ElementwiseTestCPU.*"

# 运行特定模块（CUDA）— 必须从 build/tests 目录运行！
cd build/tests
./insight_tests_cuda --gtest_filter="LinalgTestGPU.*"

# 运行单个测试
./tests/insight_tests_cpu --gtest_filter="ElementwiseTestCPU.AddSameShape"
```

```bash
# Python 绑定测试
PYTHONPATH=bindings/python LD_LIBRARY_PATH=build/backends/cpu python3 -m pytest tests/bindings/ -v

# Lua 绑定测试
LUA_PATH="bindings/lua/?/init.lua;;" LUA_CPATH="build/bindings/lua/?.so;;" LD_LIBRARY_PATH=build/backends/cpu luajit tests/bindings/test_lua_binding.lua

# Julia 绑定测试
LD_LIBRARY_PATH=build/backends/cpu julia tests/bindings/test_binding.jl

# 生成 Doxygen 文档
~/public/doxygen/bin/doxygen Doxyfile
```

> **注意**：CUDA 测试必须从 `build/tests/` 目录运行，因为动态库被复制到该目录下。机器环境：百度 AI Studio A800-SXM4-80GB。

> **Pre-commit**：提交前必须跑 clang-format，路径在 `~/.local/bin`：
> ```bash
> PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
> ```

**测试统计**：CPU 630+ 测试，CUDA 510+ 测试，全部通过

| 测试套件 | CPU | CUDA |
|----------|-----|------|
| cast | 9 | ✅ 9 |
| complex | 22 | ✅ 22 |
| creation | 27 | ✅ 27 |
| csv | 1 | ✅ 1 |
| dtype | 9 | ✅ 9（共享） |
| elementwise | 28 | ✅ 28 |
| fft | 19 | ✅ 19 |
| indexing | 41 | ✅ 33 |
| linalg | 43 | ✅ 43（15 原生 + 13 fallback + 15 其他） |
| manipulation | 42 | ✅ 42 |
| operator | 50 | ✅ 50 |
| print | 11 | ✅ 11 |
| random | 31 | ✅ 31 |
| reduction | 24 | ✅ 24 |
| signal (core) | 10 | ✅ 10（组合算子） |
| signal_windows | 30 | ✅ 30 |
| signal_waveforms | 18 | ✅ 18 |
| signal_bsplines | 13 | ✅ 13 |
| signal_filter_design | 22 | ✅ 22 |
| signal_convolution | 21 | ✅ 17 |
| signal_filtering | 23 | ✅ 15 |
| signal_spectral | 11 | — |
| signal_wavelets | 13 | — |
| signal_acoustics | 9 | — |
| signal_radar | 7 | — |
| signal_io | 11 | — |
| signal_peak_finding | 3 | — |
| signal_demod | 1 | — |
| signal_estimation | 1 | — |
| plot | 13 | — |
| unary | 27 | ✅ 27 |
| audio | 9 | ✅ 9 |
| **总计** | **630+** | **510+** |

**Python 精度对齐测试**：194 CPU + 190 CUDA（vs NumPy）

**绑定测试**：Python 76+54+384, Lua 208, Julia 74

## 添加新算子的完整流程

1. **确定算子归属模块**（elementwise, linalg, signal 等）
2. **在 `include/insight/ops/` 添加前端声明**（如果尚未添加）
3. **在 `src/ops/` 添加前端实现**（类型提升、广播、调用 launch）
4. **实现 CPU 内核** `backends/cpu/kernels/<module>/<op>.cpp`
5. **实现 CUDA 内核** `backends/cuda/kernels/<module>/<op>.cu`
6. **添加测试** `tests/cpu/test_<module>.cpp` 和 `tests/cuda/test_<module>.cpp`
7. **本地验证**：`ctest -R <op>`
8. **提交 PR**（CI 自动检查）

## 常见陷阱

1. **忘记处理非连续视图**：必须使用 strides，不能假设连续
2. **忘记 `extern "C"`**：C ABI 入口必须用 `extern "C"` 包裹
3. **忘记设置错误信息**：失败时必须调用 `cpu_set_last_error()`
4. **复数类型混用**：CUDA 用 `cuFloatComplex`，CPU 用 `std::complex<float>`，注意区分
5. **内存泄漏**：CUDA 中 `alloc_*_metadata()` 后必须 `free_*_metadata()`
6. **CUDA 测试悬垂指针（最常见！）**：`.to(CPUPlace())` 返回临时 `Array`，在语句结束时析构，后续 `data<T>()` 指针悬垂，读到垃圾数据。必须用命名变量持有：
   ```cpp
   // ❌ 错误 — 临时对象析构后 data 悬垂
   const float *data = c.to(CPUPlace()).data<float>();

   // ✅ 正确 — 命名变量持有生命周期
   Array cpu_c = c.to(CPUPlace());
   const float *data = cpu_c.data<float>();
   ```
   **症状**：测试输出出现 `1.0578383e+24`、`3.0687035e-41` 等垃圾值，或 `Device-to-host copy failed`。
   **为什么这么坑**：部分元素可能碰巧为 0（未初始化内存），看起来像 kernel 逻辑错误，实际是测试代码的内存问题。
7. **kernel 动态输出不能用 `Array(Shape({0}), ...)` 做占位符**：对于需要 kernel 内部 `cudaMalloc` 分配输出的场景（如 `unique`、`nonzero`），必须用默认构造 `Array()` + `Array(layout_ptr())` 模式，不能用显式构造的 `Array(Shape({0}), ...)`。原因是显式构造会设置 `ref_count`，干扰 `allocate_gpu_output` 的生命周期管理。参考 `nonzero()` 的实现模式。
8. **`.gitignore` 的 `*.cmake` 规则**：会阻止 `cmake/` 目录下的 `.cmake` 文件被追踪。必须加 `!cmake/*.cmake` 例外。
9. **FetchContent + apply_patch 路径**：`apply_patch` 必须放在 `if(local)/else(FetchContent)` 各分支内部，使用正确的 source dir（`_deps/<name>-src` 而非 `third_party/<name>`）。
10. **Codec deps 的 ALIAS target**：vorbis/FLAC 不创建 `Vorbis::`/`FLAC::` 命名空间 ALIAS。用 `add_subdirectory` 构建时需手动创建 ALIAS，不能用 IMPORTED target + 文件路径（会导致 `No rule to make target` 错误）。
11. **绑定 `load_backend` 必须 catch 异常**：找不到 CUDA .so 时 `ins::load_backend()` 抛异常，未捕获会崩溃进程。Lua 用 `try/catch + return bool`，Python 用 `try/catch + throw py::value_error`。
12. **测试临时目录**：用 `std::filesystem::create_directories` / `remove_all`，不要用 `system("mkdir -p ...")`。目录管理放在 `SetUpTestSuite`/`TearDownTestSuite`（suite 级别），不要放在 `SetUp`/`TearDown`（per-test 级别），避免 CI 文件系统延迟导致竞态。
13. **`write_bin` 必须显式 `ofs.close()`**：CI 文件系统可能不在 `ofstream` 析构时立即 flush，后续 `read_bin` 会读到空文件。
14. **CMake `file(GLOB)` 在 configure 时运行**：不能用 `file(GLOB)` 找 build-time 产出的文件（如 backend `.so`）。必须用 `cmake -P` 脚本在 build 时 glob + copy，或用 `$<TARGET_FILE:target>` 生成器表达式。
15. **sol2 `lua_pushcfunction` 必须 try-catch**：sol2 不总是自动捕获 C++ 异常。`__call`、`from_table`、`reshape` 等通过 `lua_pushcfunction` 注册的函数必须手动 try-catch + `luaL_error`（PR #40）。
16. **`long long` ≠ `int64_t`**：Linux x86_64 上 `int64_t` 是 `long`，`long long` 是不同类型。`dtype_of<long long>` 需要特化（PR #41）。不要加 `Shape(initializer_list<long long>)` 重载——会导致所有 `Shape({2,3})` 调用歧义。
17. **Julia reshape nullptr**：C API 返回 nullptr 时 Julia 必须检查 `C_NULL`，否则静默失败。

## 比赛评分差距分析（代码层面）

### 评分标准 vs 现状

| 评分项 | 分值 | 状态 | 说明 |
|--------|------|------|------|
| 任务1 功能实现 | 20 | ✅ | pulse_compression + pulse_doppler + ca_cfar + ambgfun |
| 任务2 功能实现 | 25 | ⚠️ | demo 用高斯核代替 B-spline、手写 Kalman 代替 KalmanFilter |
| 计算耗时 | 5 | ❌ | 需 cudaEvent 计时（HAL 已实现，demo 未调用） |
| 计算误差 | 5 | ⚠️ | --device cpu 可做对比，但缺专门的误差报告程序 |
| 稳定性 | 5 | ⚠️ | 异常输入测试已覆盖，缺内存泄漏检测 |
| 附加分 | 15 | ✅ | ~70 个附录中额外算子，远超 30 个需求 |

### 待补全 cuSignal 算子（7 个）

| 算子 | 分类 | 状态 | 实现方式 |
|------|------|------|----------|
| `lombscargle` | spectral | 后端 kernel 已有，缺前端 | 组合算子（前端 + 绑定） |
| `istft` | spectral | 完全缺失 | 组合算子（overlap-add + ifft） |
| `sosfilt` | filtering | 完全缺失 | 组合算子（级联二阶节） |
| `upfirdn` | filtering | 完全缺失 | 组合算子（upsample + firfilter + downsample） |
| `channelize_poly` | filtering | 完全缺失 | 组合算子（多相分解 + FFT） |
| `qmf` | windows | 完全缺失 | 组合算子（镜像滤波器设计） |
| `firfilter2` | filtering | 完全缺失 | 组合算子（2D FIR 滤波） |

每个算子需要：C++ 前端 API + CPU/CUDA backend kernel + Python/Lua/Julia 绑定 + CPU/CUDA 测试

## 后续优化

1. **FFT plan 缓存**：已实现 cuFFT 多 plan 缓存（unordered_map, max 16 entries）+ FFTW R2C/C2R 多 plan 缓存。C2C 仍每次重建（buffer 地址依赖）。
2. **cudaEvent 计时**：HAL `C_Event.elapsed_time` 已实现（CUDA 后端），需要暴露到前端 API 并在 demo 中使用。
3. **CPU 验证对比程序**：需要写专门的 CPU vs GPU 误差对比脚本，输出 max abs error / relative error。
4. **内存泄漏检测**：可在 demo 中加 `cudaMemGetInfo` 前后对比。
