---
name: add-cpu-op
description: 在 CPU 后端添加新算子
---

# CPU 后端算子添加规范

## 1. 确定模块

算子归属于一个模块（unary、elementwise、reduction、fft、linalg、indexing、manipulation、random、creation、signal 等）。

- 已有模块：直接在 `backends/cpu/kernels/<module>/` 下添加文件
- 新模块：创建文件夹，建议同时创建 `common.h` 提供公共宏和工具函数

## 2. 文件结构

文件路径：`backends/cpu/kernels/<module>/<op>.cpp`

```cpp
#include "common.h"           // 模块级公共头（如有）
#include "insight/c_api/array.h"

C_Status <op>_kernel_cpu(void **inputs, void **outputs) {
    // 1. 提取 InsightArray 指针
    // 2. 指针校验 → cpu_set_last_error → return C_FAILED
    // 3. switch (dtype) 分发
    //    - 使用模块宏减少代码（如 UNARY_KERNEL_LOOP）
    //    - 每个 case 处理一种 dtype
    // 4. return C_SUCCESS
}

// 注册所有支持的 dtype
REGISTER_CPU_KERNEL(<op>, INSIGHT_DTYPE_F32, <op>_kernel_cpu);
REGISTER_CPU_KERNEL(<op>, INSIGHT_DTYPE_F64, <op>_kernel_cpu);
// ...
```

## 3. 函数签名

推荐 `<op>_kernel_cpu` 命名，但不强制。只要最终被 REGISTER_CPU_KERNEL 注册即可。

不需要 `extern "C"`。

## 4. 错误处理

错误机制基于 thread_local 的 `cpu_set_last_error()` / `cpu_get_last_error()`。

规则：
- 入口校验所有输入指针
- 失败时调用 `cpu_set_last_error("op: error description")` 然后 return C_FAILED
- 成功时直接 return C_SUCCESS（不需要清除错误）

## 5. 宏使用

优先使用模块 common.h 中的宏减少代码量。常见宏：

| 宏 | 用途 |
|----|------|
| `UNARY_KERNEL_LOOP(CTYPE, Func)` | 一元操作，stride 感知，OpenMP 并行 |
| `UNARY_CMP_LOOP(CTYPE, Func)` | 返回 bool 的一元操作 |
| `ELEMENTWISE_KERNEL_LOOP(CTYPE, OP)` | 二元算术操作 |
| `ELEMENTWISE_CMP_LOOP(CTYPE, OP)` | 二元比较操作 |

如果没有现成宏可参考，可自行在 common.h 中定义，或直接展开实现。

## 6. Dtype 注册范围

根据算子类型选择注册的 dtype：

| 操作类别 | 注册类型 |
|----------|----------|
| 算术 (add, sub, mul, div) | bool + 整数(U8/I8/I16/I32/I64/U16/U32/U64) + 浮点(F32/F64) + 复数(C32/C64) |
| 比较 (equal, greater) | 同算术 |
| 逻辑 (logical_and/or/xor) | 仅 bool |
| 位运算 (bitwise_and/or/xor) | 整数 + bool |
| 移位 | 整数 |
| 一元数学 (abs, sin, exp) | 浮点 + 复数（视函数而定） |
| 规约 (sum, max) | 浮点 + 整数 |

### 特殊浮点类型

- **F16/BF16/F8_E4M3/F8_E5M2**：这些类型的支持取决于硬件环境。CPU 后端通常难以直接运算这些类型，一般仅在内存操作（复制、take 等）中按等宽整数类型（如 uint8、uint16）处理。运算类算子是否支持，开发者应结合自身硬件能力动态调整。
- 不同 GPU 架构对这些类型的支持程度不同（如 Ampere 支持 F16/BF16，Hopper 额外支持 F8），注册时按实际能力选择。

## 7. 性能优化

### OpenMP 动态阈值

不同机器（笔记本 vs 超算）核心数差异大，OpenMP 阈值不应硬编码。

**方案**：通过 `_Pragma("omp parallel for if(condition)")` 按需开启，阈值由运行时环境决定。

```cpp
// 在 backends/cpu/registry/cpu_registry.h 中定义
#include <cstdlib>
#include <omp.h>

static inline int64_t insight_omp_threshold() {
    // 环境变量优先：INSIGHT_OMP_THRESHOLD=4096
    static const char *env = std::getenv("INSIGHT_OMP_THRESHOLD");
    if (env) return std::atol(env);
    // 自动：每核至少 256 个元素才值得并行
    return 256 * omp_get_max_threads();
}
```

在宏中使用：

```cpp
#define UNARY_KERNEL_LOOP(CTYPE, Func) \
    do { \
        /* ... 准备工作 ... */ \
        int64_t n = out->numel; \
        int do_omp = (n >= insight_omp_threshold()); \
        _Pragma("omp parallel for if(do_omp)") \
        for (int64_t linear = 0; linear < n; ++linear) { \
            /* ... stride 计算 + 运算 ... */ \
        } \
    } while (0)
```

效果：
- 2 核笔记本：阈值 512，小数据不开 OpenMP
- 64 核超算：阈值 16384，避免线程调度开销吞掉并行收益
- 用户可通过 `INSIGHT_OMP_THRESHOLD` 环境变量覆盖

### 其他优化

- **内存访问模式**：尽量保证连续内存访问，利用 CPU 缓存行
- **库调用优先**：linalg 操作优先调用 OpenBLAS，fft 优先调用 FFTW3

## 8. 编译验证

```bash
cd build && cmake --build . -j$(nproc)
```

新添加的 .cpp 文件会被 CMakeLists.txt 中的 file(GLOB_RECURSE) 自动拾取，无需手动修改构建脚本。
