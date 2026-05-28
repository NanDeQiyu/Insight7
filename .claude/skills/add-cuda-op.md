---
name: add-cuda-op
description: 在 CUDA 后端添加新算子
---

# CUDA 后端算子添加规范

## 1. 确定模块

算子归属于一个模块（unary、elementwise、reduction、fft、linalg、indexing、manipulation、random、creation、signal 等）。

- 已有模块：直接在 `backends/cuda/kernels/<module>/` 下添加 `.cu` 文件
- 新模块：创建文件夹 + `common.cuh`

## 2. 文件结构

文件路径：`backends/cuda/kernels/<module>/<op>.cu`

```cpp
#include "../../registry/cuda_registry.h"
#include "common.cuh"
#include "insight/c_api/array.h"

// kernel 实现 + wrapper

REGISTER_GPU_KERNEL(<op>, INSIGHT_DTYPE_F32, <op>_kernel_gpu);
// ...
```

- 不需要 `extern "C"`
- 每个算子一个文件
- `common.cuh` 提供模块内共享的结构体、工具函数、宏

## 3. common.cuh 设计

每个模块根据自身需求在 `common.cuh` 中定义所需内容，包括但不限于：

- **Metadata 结构体** — 用于在 host 和 device 之间传递形状/步长等信息
- **辅助函数** — 如 offset 计算、launch 配置、host→device 拷贝封装
- **宏** — 如批量生成 kernel 的模板宏

不同模块的 common.cuh 内容差异很大，没有统一模板。设计时参考：
- 已有的 `elementwise/common.cuh`（二元操作，broadcast 支持）
- 已有的 `cast/common.cuh`（类型转换，按字节/类型分发）
- CPU 后端对应模块的 `common.h`（算法逻辑的参考）

## 4. Wrapper 函数

签名：`C_Status <op>_kernel_gpu(void **inputs, void **outputs)`

职责：
1. 从 `inputs` / `outputs` 提取 `InsightArray*`
2. 指针校验 → `gpu_set_last_error()` → return `C_FAILED`
3. 准备 device 端数据（分配 Metadata、拷贝参数等）
4. 计算 launch 配置（blocks/threads）
5. dispatch kernel（按 dtype switch）
6. `cudaGetLastError()` 检查
7. 释放临时 device 内存
8. return `C_SUCCESS` 或 `C_FAILED`

## 5. Kernel 实现模式

根据算子类型选择合适的实现方式：

- **自写 kernel** — 适用于 elementwise、unary、reduction、indexing 等逐元素或逐块操作
- **库调用** — 适用于 linalg（cuBLAS/cuSOLVER）、fft（cuFFT）等，wrapper 中直接调用库 API
- **混合** — 部分预处理 + 库调用 + 后处理

## 6. 错误处理

```cpp
// 自定义错误
gpu_set_last_error("op: error description");

// CUDA API 错误检查
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
}
```

## 7. Dtype 注册

使用 `REGISTER_GPU_KERNEL(op, dtype, func)` 每行注册一个 dtype。

注册范围参考 CPU 后端对应算子。GPU 可额外支持 F16/BF16（取决于硬件架构，开发者自行调整）。

## 8. 性能优化

- **Launch 配置**：线程数一般 256，block 数根据 numel 计算。大规模数据可考虑用 2D/3D grid 突破 1D grid 的 2^31-1 限制
- **内存合并访问**：kernel 中尽量保证相邻线程访问相邻内存地址
- **避免 host↔device 频繁拷贝**：Metadata 等小数据一次拷贝，不要在循环中反复 cudaMalloc/cudaFree
- **库调用优先**：linalg/fft 等有成熟库（cuBLAS/cuFFT）的场景，优先调库而非自写 kernel
- **共享内存**：需要数据复用的场景（如 reduction、tile）考虑使用 shared memory 减少全局内存访问

## 9. 编译验证

```bash
cd build && cmake --build . -j$(nproc)
```

新 `.cu` 文件会被 `file(GLOB_RECURSE)` 自动拾取，无需修改 CMakeLists.txt。
