# Insight7 - 科学计算框架

## 项目概述

Insight 是一个轻量级工业级 C++ 张量计算框架，用于信号处理和 GPU 加速。

**设计灵感**：
- **PaddlePaddle** — 算子注册机制、设备 HAL 抽象
- **Torch7** — 干净 C API、TH/THC 精神
- **NumPy/CuPy** — Python 端 API 约定

**当前状态**：
- ✅ CPU 后端完整实现（380+ 测试通过）
- 🔄 CUDA 后端正在从旧代码迁移到 C ABI 架构
- 🎯 目标：复刻 cuSignal 信号处理功能（比赛任务）

## 架构

```
insight/
├── include/insight/
│   ├── core/           # Array, Shape, Strides, DType, Place
│   ├── ops/            # 前端 API 声明 (elementwise.h, fft.h, signal.h)
│   ├── c_api/          # C ABI 接口 (array.h, kernel.h, dtype.h, place.h)
│   └── plugin/         # 算子注册 + 设备 HAL
├── src/
│   ├── core/           # Array 实现、内存管理
│   ├── ops/            # 前端算子逻辑
│   └── internal/       # 内部工具
├── backends/
│   ├── cpu/kernels/    # CPU 内核实现 (OpenMP + FFTW + OpenBLAS)
│   └── cuda/kernels/   # CUDA 内核实现 (cuBLAS + cuFFT + Thrust)
└── tests/
    ├── cpu/            # CPU 测试 (380+ tests)
    └── cuda/           # CUDA 测试 (进行中)
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

## CUDA 后端待办

### 需要移植的模块

| 模块 | 缺失算子数 | 优先级 | 参考 CPU 实现 |
|------|-----------|--------|---------------|
| **creation** | 4 (arange/eye/linspace/logspace) | 高 | `backends/cpu/kernels/creation/` |
| **fft** | 6 (fft/ifft/rfft/irfft/fft2/ifft2) | 高 | `src/ops/fft.cpp` |
| **indexing** | 17 | 中 | `backends/cpu/kernels/indexing/` |
| **linalg** | 26 | 中 | `backends/cpu/kernels/linalg/` |
| **manipulation** | 12 | 中 | `backends/cpu/kernels/manipulation/` |
| **random** | 14 | 低 | `backends/cpu/kernels/random/` |
| **reduction** | 21 | 中 | `backends/cpu/kernels/reduction/` |
| **unary** | 38 | 高 | `backends/cpu/kernels/unary/` |

### 移植优先级

1. **高优先级**：creation, fft, unary（比赛任务核心）
2. **中优先级**：linalg, reduction, manipulation
3. **低优先级**：random, indexing

## 测试

```bash
# 运行所有测试
cd build && ctest

# 运行特定模块
./tests/insight_tests_cpu --gtest_filter="ElementwiseTestCPU.*"

# 运行单个测试
./tests/insight_tests_cpu --gtest_filter="ElementwiseTestCPU.AddSameShape"
```

**测试统计**：380+ 测试，16 个测试套件，全部通过

| 测试套件 | 测试数 | CPU | CUDA |
|----------|--------|-----|------|
| dtype | 9 | ✅ | N/A |
| cast | 8 | ✅ | ✅ |
| creation | 10+ | ✅ | 🔄 |
| elementwise | 20+ | ✅ | ✅ |
| unary | 20+ | ✅ | 🔄 |
| random | 30+ | ✅ | 🔄 |
| indexing | 20+ | ✅ | 🔄 |
| manipulation | 20+ | ✅ | 🔄 |
| reduction | 20 | ✅ | 🔄 |
| fft | 19 | ✅ | 🔄 |
| linalg | 30+ | ✅ | 🔄 |
| signal | 10 | ✅ | ❌（待实现）|

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
