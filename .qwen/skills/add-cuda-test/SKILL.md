---
name: add-cuda-test
description: Workflow for adding CUDA tests aligned with CPU tests, including common pitfalls and runtime fixes.
source: auto-skill
extracted_at: '2026-05-29T16:25:13.968Z'
updated: '2026-06-01'
---

# Workflow: Add CUDA Tests

When adding new CUDA tests to align with existing CPU tests, follow this workflow to ensure proper build and discovery.

## 1. Create Test File

- Create `tests/cuda/test_<module>.cpp`.
- Use `file(GLOB)` in `tests/CMakeLists.txt` to automatically pick up new files.

### Signal Module Tests (all 14 submodules have backend kernels)

All 14 signal submodules now have dedicated CPU and CUDA backend kernels with
the `signal_` naming prefix convention. Signal CUDA tests mirror CPU tests exactly:

```
tests/cpu/test_signal_convolution.cpp    → tests/cuda/test_signal_convolution.cpp
tests/cpu/test_signal_filtering.cpp      → tests/cuda/test_signal_filtering.cpp
tests/cpu/test_signal_windows.cpp        → tests/cuda/test_signal_windows.cpp
... (one pair per submodule)
```

Signal kernel naming convention:
- Kernel function: `signal_<op>_kernel_cpu` / `signal_<op>_kernel_gpu`
- REGISTER macro: `REGISTER_CPU_KERNEL(signal_<op>, dtype, func)`
- Example: `signal_exponential`, `signal_square`, `signal_morlet`, `signal_lombscargle`
- The `signal_` prefix avoids collisions with non-signal kernels (e.g., `exponential` in random module)

Signal dtype support:
- **CPU kernels**: F64, F32
- **CUDA kernels**: F64, F32, F16, BF16
- OpenMP parallelism for CPU when `numel > 1000`
- 256 threads/block for CUDA

### Composite Operations (non-signal modules)

Some modules are implemented as **composite operations** — they call other primitives
(`diff`, `sin`, `rfft`, etc.) that already have CUDA kernels. For these modules:

- No `backends/cuda/kernels/<module>/` directory is needed
- Only the CUDA test file needs to be created
- The operations work on GPU arrays automatically through the underlying primitives
- Use the same test logic as CPU, just add GPU data transfer

Example: `unwrap` calls `diff`, `greater_than`, `abs`, `where`, `cumsum`, `pad`, `add` — all of which have CUDA kernels.

## 2. GPU Fixture Pattern — CRITICAL

**必须使用 `ins::init({"cpu", "cuda"})` 而不是 `ins::init()`！**

空参数的 `ins::init()` 不会加载 CUDA 后端，导致所有测试被 skip。

```cpp
class <Module>TestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "cuda"});  // ← 必须显式传入后端列表
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
  }
};
```

## 3. Data Transfer
- **CPU to GPU**: `Array gpu_arr = cpu_arr.to(GPUPlace(0));`
- **GPU to CPU**: `Array cpu_arr = gpu_arr.to(CPUPlace());`
- **Pattern**: Create data on CPU, transfer to GPU, run operation, transfer result back to CPU for verification.

## 4. Build & Verify

### 4.1 CMake 配置（用户指定的命令）
```bash
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$HOME/public/OpenBLAS" \
    -DINSIGHT_WITH_CUDA=ON \
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON \
    -DINSIGHT_BUILD_TESTS=ON
```

### 4.2 新文件必须重新 cmake
`file(GLOB)` 在 cmake configure 阶段缓存文件列表。添加新测试文件后，必须重新运行 cmake 才能发现新文件，然后 make：
```bash
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$HOME/public/OpenBLAS" \
    -DINSIGHT_WITH_CUDA=ON \
    -DINSIGHT_USE_FFTW3=ON \
    -DINSIGHT_USE_OPENBLAS=ON \
    -DINSIGHT_BUILD_TESTS=ON
make -j$(nproc)
```

### 4.3 运行测试 — 必须从 tests 目录！
动态库 (`libinsight_cuda_backend.so`) 被复制到 `tests/` 目录，不是 `build/` 目录。
```bash
cd build/tests
./insight_tests_cuda --gtest_filter="<Module>TestGPU.*"
```

## 5. Common Pitfalls

### 5.1 BOOL dtype not supported in CUDA kernel
Many CUDA kernels don't register `INSIGHT_DTYPE_BOOL`. If a test uses `bool` arrays (e.g., masks), the kernel may fail with "kernel not found". Fix the kernel by adding BOOL registration, or cast to `F32` in the test as a workaround.

**bitwise_not on BOOL**: For `~bool`, use `!` (logical not) instead of `~` (bitwise not) in the CUDA kernel. On CUDA, `~true` = `~1` = `0xFE` which is still truthy — wrong behavior. Match NumPy/PyTorch/PaddlePaddle where `~bool` is logical not:

```cpp
// CUDA kernel: separate BOOL specialization
__global__ void bitwise_not_bool_kernel(const bool *x, bool *out, ...) {
  out[idx] = !x[idx];  // logical not, NOT bitwise not
}

// Register
REGISTER_GPU_KERNEL(bitwise_not, INSIGHT_DTYPE_BOOL, bitwise_not_kernel_gpu);
```

The CPU kernel already does this: `UNARY_KERNEL_LOOP(bool, [](bool v) { return !v; });`

### 5.2 CUDA kernel naming mismatches with CPU
Some CUDA kernels register with different names than the CPU version:
- CPU: `negative` → CUDA: `neg` (已修复为 `negative`)
- Check `REGISTER_GPU_KERNEL(name, ...)` vs `REGISTER_CPU_KERNEL(name, ...)` for mismatches.

### 5.3 CUDA kernel dtype registration gaps
Many CUDA kernels only register `F32`/`F64`, while CPU kernels support `I32`/`I64` etc.
- If a test uses `DType::I32` but the CUDA kernel only supports `F32`, you'll get "kernel not found"
- Fix: either add dtype registration to the kernel, or change the test to use `F32`

### 5.4 Helper functions should be `_gpu` suffixed
Define GPU-specific helpers with `_gpu` suffix to avoid conflicts with CPU helpers:
```cpp
template <typename T> void fill_sequential_gpu(Array &gpu_arr) {
  Array cpu_arr(gpu_arr.shape(), gpu_arr.dtype(), CPUPlace());
  T *data = cpu_arr.data<T>();
  int64_t n = cpu_arr.numel();
  for (int64_t i = 0; i < n; ++i) data[i] = static_cast<T>(i);
  gpu_arr = cpu_arr.to(GPUPlace(0));
}
```

### 5.5 Test names must match CPU exactly
For easy comparison, use the exact same test names as the CPU version, just with `GPU` suffix on the fixture class.

### 5.6 Kernels with dynamic output allocation
Some kernels (`nonzero`, `unique`) allocate their own output arrays. If you need to call these in tests, make sure the kernel properly allocates GPU memory and sets `InsightArray` metadata. See `fix-cuda-kernel-output-allocation` skill for details.

### 5.7 Dangling pointer from `.to(CPUPlace()).data<T>()`
**NEVER** use `c.to(CPUPlace()).data<T>()` inline — the temporary `Array` is destroyed at the end of the statement, leaving a dangling pointer. Always use a named variable:
```cpp
// BAD — dangling pointer
const float *data = c.to(CPUPlace()).data<float>();

// GOOD — named variable
Array cpu_c = c.to(CPUPlace());
const float *data = cpu_c.data<float>();
```
See `fix-test-dangling-pointer` skill for full details.

### 5.8 Array creation must use CPUPlace() for memcpy
When the current device is GPU (after `set_device(GPUPlace(0))`), `Array(Shape(...), DType)` allocates on GPU. `std::memcpy` can't write to GPU memory → segfault.

```cpp
// WRONG — segfault when current device is GPU
Array result(Shape({rows, cols}), DType::F64);
std::memcpy(result.data<double>(), data.data(), ...);

// CORRECT — always specify CPUPlace() for host-side data initialization
Array result(Shape({rows, cols}), DType::F64, CPUPlace());
std::memcpy(result.data<double>(), data.data(), ...);
Array gpu_result = result.to(GPUPlace(0));
```

### 5.9 `to_array()` defaults to `get_device()` — same GPU trap as 5.8

`to_array(std::vector<T>)` uses `get_device()` as the default `place` parameter. In CUDA tests, `SetUpTestSuite` calls `set_device(GPUPlace(0))`, so `get_device()` returns `GPUPlace(0)`. This means `to_array(data)` creates the array **on GPU**, not CPU.

Accessing `.data<T>()` on such an array from CPU code gives a GPU pointer → **segfault**.

```cpp
// BUG — to_array() puts data on GPU because get_device() == GPUPlace(0)
Array orig = to_array(original);  // GPU array!
const double *p = orig.data<double>();  // GPU pointer
EXPECT_NEAR(p[0], expected, 1e-6);  // SEGFAULT: CPU dereferences GPU pointer

// FIX option 1: compare against the original std::vector directly
EXPECT_NEAR(rec_data[i], original[i], 1e-6);

// FIX option 2: explicitly create on CPU
Array orig = to_array(original, DType::F64, CPUPlace());
```

**Symptom**: Segfault when accessing `.data<T>()` on an array created by `to_array()`. The pointer looks valid (non-null, page-aligned address like `0x7f...`) but dereferencing it crashes. This is very misleading — it looks like a GPU memory corruption bug but is actually a CPU-side access to a GPU pointer.

## 6. Run Tests
```bash
cd build/tests
./insight_tests_cuda --gtest_filter="<Module>TestGPU.*"
```
