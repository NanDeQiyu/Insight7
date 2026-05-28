# add-api.md — 新增前端 API 流程

前端 API 是用户直接调用的函数（如 `add()`, `sin()`），负责类型提升、广播、设备统一，然后调度到后端 kernel。

## 文件位置

- 头文件：`include/insight/ops/<module>.h`
- 实现：`src/ops/<module>.cpp`

## 重要规则

1. **注释必须用英文 Doxygen 格式**（`/** */` 风格，含 `@brief`, `@param`, `@return`）
2. **新增 API 后必须立即写测试**，不能留到后面补

## 流程总览

```
1. include/insight/ops/<module>.h   — 声明函数（带 Doxygen 注释）
2. src/ops/<module>.cpp             — 实现调度逻辑
3. backends/cpu/kernels/<module>/   — CPU kernel（见 add-cpu-op.md）
4. backends/cuda/kernels/<module>/  — CUDA kernel（见 add-cuda-op.md）
5. tests/cpu/test_<module>.cpp      — 测试（见 test-op.md）  ← 必须立即写
```

## 操作类型与模板

### 1. 一元操作（最简单）

适用于：`abs`, `sin`, `exp`, `floor`, `sign`, `isnan` 等

**头文件**（`include/insight/ops/elementwise.h`）：
```cpp
/**
 * @brief Compute absolute value element-wise.
 *
 * @param x Input array
 * @return Array with same shape and dtype as x
 */
Array abs(const Array &x);
```

**实现**（`src/ops/unary.cpp`）— 使用宏批量定义：
```cpp
DEFINE_UNARY_OP(abs);
```

`DEFINE_UNARY_OP` 宏展开为：
```cpp
Array abs(const Array &x) {
  INS_CHECK(x.defined(), "abs: input is undefined");
  DType out_dtype = x.dtype();
  // logical_not 等特殊算子输出 bool
  if (std::string("abs") == "logical_not") out_dtype = DType::BOOL;
  Array out(x.shape(), out_dtype, x.place());
  ops().launch("abs", x.place(), x.dtype(), {(void *)x.layout_ptr()},
               {out.layout_ptr()});
  return out;
}
```

**新增步骤**：
1. 在头文件加声明
2. 在 `src/ops/unary.cpp` 加 `DEFINE_UNARY_OP(your_op);`
3. 实现 CPU kernel

### 2. 二元操作

适用于：`add`, `sub`, `mul`, `div`, `pow`, `mod`, `maximum`, `minimum`

**头文件**（`include/insight/ops/elementwise.h`）：
```cpp
/**
 * @brief Element-wise addition of two arrays.
 *
 * Supports broadcasting and type promotion.
 *
 * @param a First input array
 * @param b Second input array
 * @return Array containing a + b
 */
Array add(const Array &a, const Array &b);
```

**实现**（`src/ops/elementwise.cpp`）— 使用通用调度器：
```cpp
Array add(const Array &a, const Array &b) { return binary_op(a, b, "add"); }
```

`binary_op` 模板自动处理：
1. 类型提升：`promote_types(a.dtype(), b.dtype())`
2. 类型转换：`a.to(out_dtype)`
3. 设备统一：`promote_places(a.place(), b.place())`
4. 广播：`broadcast_arrays({a, b})`
5. 分配输出：`Array out(a1.shape(), out_dtype, target_place)`
6. 调度：`ops().launch("add", ...)`

**新增步骤**：
1. 在头文件加声明
2. 在 `src/ops/elementwise.cpp` 加一行：`Array xxx(...) { return binary_op(a, b, "xxx"); }`
3. 实现 CPU kernel

### 3. 比较操作

适用于：`equal`, `greater`, `less`, `greater_equal`, `less_equal`

**实现**— 使用 `cmp_op` 调度器：
```cpp
Array equal(const Array &a, const Array &b) { return cmp_op(a, b, "equal"); }
```

与 `binary_op` 的区别：输出类型固定为 `DType::BOOL`，用 `common_dtype` 分发 kernel。

### 4. 逻辑操作

适用于：`logical_and`, `logical_or`, `logical_xor`

**实现**— 使用 `logical_op` 调度器：
```cpp
Array logical_and(const Array &a, const Array &b) {
  return logical_op(a, b, "logical_and");
}
```

特点：输入先转 bool，输出也是 bool。

### 5. 创建操作

适用于：`zeros`, `ones`, `full`, `eye`, `arange`, `linspace`

**头文件**（`include/insight/ops/creation.h`）：
```cpp
Array zeros(const Shape &shape, DType dtype = DType::F32,
            const Place &place = get_device());
```

**实现**（`src/ops/creation.cpp`）：
```cpp
Array full(const Shape &shape, double fill_value, DType dtype, const Place &place) {
  Array result(shape, dtype, place);
  ops().launch("full", place, dtype, {result.layout_ptr(), &fill_value},
               {result.layout_ptr()});
  return result;
}
```

特点：没有输入数组，只有输出 + 参数。参数通过 `void**` 传递。

### 6. 规约操作

适用于：`sum`, `mean`, `max`, `min`, `argmax`, `cumsum`

**头文件**（`include/insight/ops/reduction.h`）：
```cpp
Array sum(const Array &x, std::optional<int> axis = std::nullopt, bool keepdim = false);
```

**实现**特点：
- 用 `ReductionInfo` 结构体管理轴信息
- 用 `transpose` 把规约轴移到最后一维
- 调用 `ops().launch()` 后恢复轴顺序
- `axis=nullopt` 表示全规约（flatten）

### 7. 组合操作

适用于：`sinc`, `unwrap`, `convolve`

**实现**— 用已有 API 组合：
```cpp
Array sinc(const Array &x) {
  Array pi_x = x * M_PI;
  Array numerator = sin(pi_x);
  // ... 组合 elementwise, creation, where 等
  return result;
}
```

特点：不直接调用 `ops().launch()`，而是组合已有前端 API。

## 参数传递规范

| 参数类型 | 传递方式 | 示例 |
|----------|----------|------|
| 输入数组 | `a.layout_ptr()` | `{a.layout_ptr(), b.layout_ptr()}` |
| 输出数组 | `out.layout_ptr()` | `{out.layout_ptr()}` |
| 标量参数 | `&value`（取地址） | `{result.layout_ptr(), &fill_value}` |
| 整数参数 | `&k` | `{result.layout_ptr(), &k}` |
| 多个标量 | 依次取地址 | `{..., &start, &stop, &base}` |

## 类型提升规则

使用 `promote_types()`（在 `insight/utils/promotion.h`）：
- 整数 + 浮点 → 浮点
- F32 + F64 → F64
- 实数 + 复数 → 复数
- 任何类型 + bool → 该类型

## 设备统一规则

使用 `promote_places()`：
- CPU + CPU → CPU
- GPU + GPU → GPU
- CPU + GPU → GPU（数据搬到 GPU）

## 输入检查

使用 `INS_CHECK` 宏：
```cpp
INS_CHECK(x.defined(), "func_name: input is undefined");
INS_CHECK(x.shape().ndim() == 2, "func_name: input must be 2D");
INS_CHECK(ax >= 0 && ax < ndim, "func_name: axis out of range");
```

## 完整示例：新增 `hypot` 算子

`hypot(a, b) = sqrt(a² + b²)`

### 方案 A：组合实现（推荐，简单）

1. **头文件**（`include/insight/ops/elementwise.h`）：
```cpp
/**
 * @brief Compute the hypotenuse: sqrt(a^2 + b^2).
 *
 * @param a First input array
 * @param b Second input array
 * @return Array containing sqrt(a^2 + b^2)
 */
Array hypot(const Array &a, const Array &b);
```

2. **前端**（`src/ops/elementwise.cpp`）：
```cpp
Array hypot(const Array &a, const Array &b) {
  return sqrt(add(square(a), square(b)));
}
```

3. **测试**（`tests/cpu/test_elementwise.cpp`）— 必须立即写：
```cpp
TEST_F(ElementwiseTestCPU, HypotSameShape) {
  Array a({3}, DType::F32);
  Array b({3}, DType::F32);
  float *a_data = a.data<float>();
  float *b_data = b.data<float>();
  a_data[0] = 3.0f; b_data[0] = 4.0f;  // hypot = 5
  a_data[1] = 5.0f; b_data[1] = 12.0f; // hypot = 13
  a_data[2] = 8.0f; b_data[2] = 15.0f; // hypot = 17

  Array c = hypot(a, b);

  std::vector<float> expected = {5.0f, 13.0f, 17.0f};
  expect_float_equal<float>(c, expected);
}
```

### 方案 B：独立 kernel（性能更好）

1. **头文件** — 同上（带 Doxygen 注释）

2. **前端**（`src/ops/elementwise.cpp`）：
```cpp
Array hypot(const Array &a, const Array &b) { return binary_op(a, b, "hypot"); }
```

3. **CPU kernel**（`backends/cpu/kernels/elementwise/hypot.cpp`）：
```cpp
#include "../../registry/cpu_registry.h"
#include "common.h"
#include "insight/c_api/array.h"
#include <cmath>

extern "C" {
C_Status hypot_kernel_cpu(void** inputs, void** outputs) {
    // ... 实现
}
}

REGISTER_CPU_KERNEL(hypot, INSIGHT_DTYPE_F32, hypot_kernel_cpu);
```

4. **测试** — 必须立即写，同上
