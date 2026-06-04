---
name: port-cusignal-gpu-composite
description: Composite 操作实现 GPU 兼容的关键规则 — 禁止 raw pointer writes，必须用 Insight7 API 组合
source: auto-skill
extracted_at: '2026-05-30T00:00:00.000Z'
updated: '2026-06-01'
---

# GPU-Compatible Composite Operations for Signal Porting

## 当前状态 (2026-06-01)

所有 14 个 signal 子模块已完成后端 kernel 注册（`signal_` 前缀）。
部分模块（如 demod）仍为纯 composite 实现，其余模块使用 composite 前端 + 后端 kernel 混合模式。
以下规则仍然适用于 composite 部分和新 signal 函数的开发。

## 核心规则

**所有 signal 函数必须通过 Insight7 的 composite API 实现，禁止直接写指针。**

HAL 会自动将 `arange/sin/cos/exp/where/add/sub/mul/div/full/ones/abs` 等操作
调度到 CPU 或 CUDA。直接写 `Array` 的 `data<T>()` 指针只在 CPU 上工作，GPU 上 segfault。

## 禁止的模式（会导致 GPU segfault）

```cpp
// ❌ 错误 — 在 GPU 上 data<double>() 返回 GPU 指针，CPU 写入 segfault
Array result({N}, DType::F64);
double *w = result.data<double>();
for (int64_t i = 0; i < N; ++i) {
    w[i] = some_formula(i);  // GPU 上这是 CPU→GPU 非法写入
}
```

## 正确的模式

### 模式 1: Composite 操作组合（首选）

```cpp
// ✅ 正确 — 所有操作通过 HAL 自动支持 CPU/CUDA
Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
Array phase = mul(i_arr, full({N}, 2.0 * M_PI / (N - 1), DType::F64));
Array result = add(full({N}, 0.54, DType::F64),
                   mul(full({N}, -0.46, DType::F64), cos(phase)));
```

### 模式 2: where 条件选择

```cpp
// ✅ 分段函数用 where + 比较操作
Array mask = less_than(ax, full({N}, 1.0, DType::F64));
Array mask2 = less_than(ax, full({N}, 2.0, DType::F64));
Array result = where(mask, region1, where(mask2, region2, zero));
```

### 模式 3: CPU 计算 + to_array 转移（复杂算法的 fallback）

```cpp
// ✅ 当算法太复杂无法用 composite 表达时（如镜像逻辑、Bessel 函数、Chebyshev 多项式）
return apply_sym(M, sym, [&](int64_t N) {
    std::vector<double> vals(N);
    for (int64_t i = 0; i < N; ++i) {
        vals[i] = complex_formula(i, N);
    }
    return to_array(vals);  // to_array 自动放到当前设备（CPU 或 GPU）
});
```

### 模式 4: 需要直接指针访问时，先 CPU 创建再转移

```cpp
// ✅ 如 unit_impulse 需要写单个元素
Array result = zeros(shape, dtype, CPUPlace());  // 强制 CPU
result.data<double>()[idx] = 1.0;               // CPU 指针安全写入
if (place.kind() != DeviceKind::CPU) {
    result = result.to(place);                   // 转移到目标设备
}
```

## apply_sym 辅助函数

窗函数普遍需要 `sym` 参数支持（symmetric vs periodic/DFT-even）。推荐的辅助函数：

```cpp
template <typename Fn>
Array apply_sym(int64_t M, bool sym, Fn &&fn) {
    INS_CHECK(M >= 1, "window: M must be >= 1");
    if (M == 1) return full({1}, 1.0, DType::F64);
    int64_t M_work = sym ? M : M + 1;
    Array result = fn(M_work);
    if (!sym) result = slice(result, {0}, {0}, {M});
    return result;
}
```

`fn` 接收 M_work（对称时=M，周期时=M+1），返回 Array。apply_sym 自动截断。

## 可用的 Composite 构建块

| 类别 | 操作 |
|------|------|
| 创建 | `zeros`, `ones`, `full`, `arange`, `to_array` |
| 算术 | `add`, `sub`, `mul`, `div`, `negative` |
| 数学 | `sin`, `cos`, `exp`, `log`, `abs`, `sqrt` |
| 比较 | `less_than`, `greater_than`, `less_equal`, `greater_equal` |
| 条件 | `where` |
| 规约 | `sum`, `mean`, `max`, `min` |
| 变换 | `slice`, `reshape`, `concat`, `pad` |
| FFT | `fft::fft`, `fft::ifft`, `fft::rfft`, `fft::irfft` |

## 已知陷阱

1. **`acosh(x)` 要求 x >= 1** — 对负 x，用 `acosh(|x|)` + 符号处理
2. **`to_array` 放置位置** — 跟随 `get_device()`，CUDA 测试中默认放 GPU
3. **镜像/对称逻辑** — 用 composite 难以正确表达时，直接用 CPU 计算 + `to_array`
4. **Bessel 函数等特殊数学** — 没有 composite 等价物，用 CPU 计算
5. **新增测试文件需要重新 cmake** — `file(GLOB)` 在 configure 时缓存

## CUDA 测试对齐模式

每个 CPU 测试必须有对应的 CUDA 测试，模式：

```cpp
class SignalXxxTestGPU : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        ins::init({"cpu", "cuda"});
        try { set_device(GPUPlace(0)); }
        catch (...) { GTEST_SKIP() << "GPU not available"; }
    }
};

TEST_F(SignalXxxTestGPU, SomeTest) {
    std::vector<double> data = {...};
    Array cpu_arr = to_array(data);      // CPU 创建
    Array gpu_arr = cpu_arr.to(GPUPlace(0));  // 转 GPU
    Array result = ins::signal::xxx(gpu_arr);  // GPU 计算
    Array result_cpu = result.to(CPUPlace());  // 转回 CPU
    const double *d = result_cpu.data<double>();// 命名变量持有生命周期
    EXPECT_NEAR(d[0], expected, tol);
}
```

**关键**: `.to(CPUPlace())` 必须存到命名变量，不能链式调用 `.data<T>()`（悬垂指针）。
