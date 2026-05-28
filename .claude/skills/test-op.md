# test-op.md — 算子测试编写规范

测试的是完整链路：前端 API → 调度 → kernel → 结果验证。不是单独测 kernel。

## 文件位置

- CPU: `tests/cpu/test_<module>.cpp`
- CUDA: `tests/cuda/test_<module>.cpp`

## 测试夹具

```cpp
class <Module>TestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};
```

CUDA 夹具改用 `GPUPlace(0)`。如果依赖可选库，在 SetUpTestSuite 中跳过：

```cpp
if (!ins::is_compiled_with_fftw3()) {
  GTEST_SKIP() << "FFTW3 not available";
}
```

## Helper 函数（每个文件独立定义，不抽公共头文件）

### CPU 版本

```cpp
namespace {
template <typename T> void fill_sequential(Array &arr) {
  T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    data[i] = static_cast<T>(i);
}

template <typename T> void fill_constant(Array &arr, T val) {
  T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    data[i] = val;
}

template <typename T>
void expect_equal(const Array &arr, const std::vector<T> &expected) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    EXPECT_EQ(data[i], expected[i]);
}

template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    EXPECT_NEAR(data[i], expected[i], tol);
}

void expect_bool_equal(const Array &arr, const std::vector<bool> &expected) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const bool *data = arr.data<bool>();
  for (int64_t i = 0; i < arr.numel(); ++i)
    EXPECT_EQ(data[i], expected[i]);
}

template <typename T>
void expect_complex_equal(const Array &arr,
                          const std::vector<std::complex<T>> &expected,
                          T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const std::complex<T> *data = arr.data<std::complex<T>>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i].real(), expected[i].real(), tol);
    EXPECT_NEAR(data[i].imag(), expected[i].imag(), tol);
  }
}
} // namespace
```

### CUDA 版本（加 `_gpu` 后缀，内部转 CPU 比较）

```cpp
template <typename T> void fill_sequential_gpu(Array &gpu_arr) {
  Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
  fill_sequential<T>(cpu_arr);
  gpu_arr = cpu_arr.to(GPUPlace(0));
}

template <typename T>
void expect_float_equal_gpu(const Array &gpu_arr,
                            const std::vector<T> &expected, T tol = 1e-6) {
  Array cpu_arr = gpu_arr.to(CPUPlace());
  expect_float_equal<T>(cpu_arr, expected, tol);
}
```

## 精度容差

| 类型 | 默认容差 |
|------|----------|
| F32 | 1e-5 |
| F64 | 1e-6 |
| C32/C64 | 1e-6 |

## 测试分类覆盖

每个算子的测试应覆盖以下维度（按需选择）：

1. **基本功能** — SameShape，最简单的正确性验证
2. **广播** — Broadcast，2D+1D、scalar 等
3. **非连续视图** — View/NonContiguous，用 `slice()` 产生非连续数组
4. **多 dtype** — 至少覆盖该算子注册的所有 dtype（见下方规则）
5. **复数** — 如果算子支持复数类型，必须测试 C32 和 C64
6. **边界值** — 空数组、NaN、Inf、零值等
7. **axis 操作** — 不同 axis、负 axis
8. **keepdim** — 如果算子有 keepdim 参数

## dtype 覆盖规则

**每个算子必须测试它注册的所有 dtype**。查看 `REGISTER_CPU_KERNEL` 或 `REGISTER_GPU_KERNEL` 确定支持范围。

最低覆盖要求：

| 操作类别 | 必测 dtype |
|----------|-----------|
| 算术 (add/sub/mul/div) | F32, F64, I32, I64, C32, C64 |
| 比较 (equal/greater/less) | F32, I32, BOOL |
| 逻辑 (and/or/xor) | BOOL |
| 位运算 | I32, I64 |
| 一元浮点 (abs/sin/exp) | F32, F64 |
| 一元复数 (conj) | C32, C64 |
| 规约 (sum/mean) | F32, F64, I32 |
| 创建 (zeros/ones/arange) | F32, F64, I32, I64 |

## 测试组织

- 使用 `// ============` 分隔符按功能分组
- 测试名清晰描述场景：`SubSameShape`, `Broadcast2D1D`, `ViewAdd`
- 一个文件对应一个模块，不拆分多个文件

## GPU 测试数据模式

```cpp
// 创建：CPU 填数据 → 转 GPU
Array a_cpu({2, 3}, DType::F32, CPUPlace());
fill_sequential<float>(a_cpu);
Array a = a_cpu.to(GPUPlace(0));

// 验证：结果转回 CPU
Array cpu_c = c.to(CPUPlace());
const float *data = cpu_c.data<float>();
EXPECT_FLOAT_EQ(data[0], expected_val);
```

## 构建

测试文件用 `file(GLOB)` 自动拾取，不需要改 CMakeLists.txt。添加新测试文件后重新 cmake 即可。
