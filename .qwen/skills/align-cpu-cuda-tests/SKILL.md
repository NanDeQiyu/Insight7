---
name: align-cpu-cuda-tests
description: Workflow for aligning CPU and CUDA test suites to have identical test names and counts per module.
source: auto-skill
extracted_at: '2026-05-29T16:41:31.083Z'
updated: '2026-06-01'
---

# Workflow: Align CPU and CUDA Tests

确保 CPU 和 CUDA 测试套件的测试名称和数量完全一致（排除 CPU 专属基础设施测试）。

## 0. Signal Test Alignment Status (2026-06-01)

All 14 signal submodules have aligned CPU/CUDA test suites:
- CPU tests: `tests/cpu/test_signal_<module>.cpp`
- CUDA tests: `tests/cuda/test_signal_<module>.cpp`
- All signal functions use backend kernels with `signal_` prefix convention
- Signal dtype support: CPU (F64, F32), CUDA (F64, F32, F16, BF16)
- Signal CUDA tests use the same patterns as other modules (see section 4)

Signal test file pairs:
| Submodule | CPU Tests | CUDA Tests |
|-----------|-----------|------------|
| windows | 30 | 30 |
| waveforms | 18 | 18 |
| bsplines | 13 | 13 |
| filter_design | 22 | 22 |
| convolution | 21 | 17 |
| filtering | 23 | 15 |
| spectral | 11 | — |
| wavelets | 13 | — |
| acoustics | 9 | — |
| peak_finding | 3 | — |
| demod | 1 | — |
| estimation | 1 | — |
| radar | 7 | — |
| io | 11 | — |

Note: Some signal submodules (spectral, wavelets, acoustics, etc.) currently have
CPU tests only — CUDA tests may be added in future work.

## 1. 对比测试套件

```bash
# 列出所有 CPU 测试套件
cd build && ./tests/insight_tests_cpu --gtest_list_tests 2>&1 | grep "^[A-Z]"

# 列出所有 CUDA 测试套件
cd build/tests && ./insight_tests_cuda --gtest_list_tests 2>&1 | grep "^[A-Z]"
```

## 2. 逐套件对比测试名称

对每个对齐的套件对（如 `LinalgTest` vs `LinalgTestGPU`），提取并对比测试名称：

```bash
# CPU 测试名称
./tests/insight_tests_cpu --gtest_filter="LinalgTest.*" --gtest_list_tests 2>&1 | grep "  "

# CUDA 测试名称
./insight_tests_cuda --gtest_filter="LinalgTestGPU.*" --gtest_list_tests 2>&1 | grep "  "
```

**对齐规则**：测试名称必须完全一致（忽略套件前缀）。例如 CPU 的 `Pinv2x2` 在 CUDA 中也必须是 `Pinv2x2`，不能是 `PinvFallback`。

## 3. 分类差异

| 类型 | 处理方式 |
|------|---------|
| CPU 有但 CUDA 缺少 | 添加到 CUDA（适配 GPU 模式） |
| CUDA 有但 CPU 没有 | 从 CUDA 删除，或添加到 CPU |
| 名称不同但功能相同 | 统一名称 |

## 4. 适配 CPU 测试到 CUDA

将 CPU 测试迁移到 CUDA 时，遵循以下转换模式：

```cpp
// CPU 版本
Array A = create_matrix_f64(2, 2, {1, 2, 3, 4});
Array result = some_op(A);
const double *data = result.data<double>();
EXPECT_NEAR(data[0], expected, 1e-6);

// CUDA 版本
Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});  // helper 放 CPU → GPU
Array result = some_op(A);
Array cpu_result = result.to(CPUPlace());              // 必须显式转回 CPU
const double *data = cpu_result.data<double>();
EXPECT_NEAR(data[0], expected, 1e-5);                 // 容差适当放宽
```

**关键差异**：
- helper 函数必须先在 CPUPlace 创建，再 `.to(GPUPlace(0))`
- 结果必须 `.to(CPUPlace())` 后才能用 `.data<T>()` 访问
- 浮点容差通常比 CPU 版本放宽 10-100x（GPU 计算精度差异）
- 使用 `check_matrix_equal` helper 做矩阵比较（内部已处理 to CPU）

## 5. C_FALLBACK 测试处理

C_FALLBACK 操作（如 `cond`, `pinv`, `matrix_rank` 等）在 CUDA 上走 CPU fallback 路径。
测试名称应与 CPU 一致（如 `Cond`），不要用 `CondFallback` 这样的变体名。

## 6. 验证

```bash
# 逐套件对比数量
for suite in CastTest CreationTest ...; do
  cpu=$(./tests/insight_tests_cpu --gtest_list_tests 2>&1 | awk "/^${suite}CPU?\./{found=1;next} /^[A-Z]/{found=0} found && /^  [A-Z]/{c++} END{print c}" c=0)
  cuda=$(./insight_tests_cuda --gtest_list_tests 2>&1 | awk "/^${suite}GPU?\./{found=1;next} /^[A-Z]/{found=0} found && /^  [A-Z]/{c++} END{print c}" c=0)
  echo "$suite: CPU=$cpu CUDA=$cuda"
done

# 全量运行确认通过
./tests/insight_tests_cpu 2>&1 | tail -3
./insight_tests_cuda 2>&1 | tail -3
```

## 7. 基础设施测试套件的 CUDA 迁移

以下套件虽然不涉及重计算，但需要 CUDA 版本来验证 GPU 后端的完整性：

| 套件 | CUDA 迁移要点 |
|------|--------------|
| `ComplexTest` | `to_complex`/`as_complex` 等是视图操作，在 GPU 上共享内存。结果需 `.to(CPUPlace())` 后验证。 |
| `OperatorTest` | C++ 运算符（`+`, `-`, `*`, `/`, `==` 等）会根据数组设备自动分发到 GPU kernel。注意 `~bool` 在 CUDA 上需要 `bitwise_not` BOOL kernel。 |
| `PrintTest` | `to_string()` 和 `operator<<` 需要能处理 GPU 数组。注意非连续 GPU 视图在 `to(CPUPlace())` 时可能丢失 stride 信息。 |
| `CsvTest` | `export_support_csv(filename, DeviceKind::GPU)` 导出 CUDA 算子支持矩阵。 |

**通用迁移模式**：
1. fixture 用 `ins::init({"cpu", "cuda"})` + `set_device(GPUPlace(0))`
2. 数据先在 CPUPlace 创建，再 `.to(GPUPlace(0))`
3. 结果 `.to(CPUPlace())` 后用 `.data<T>()` 访问
4. 比较时可用原始 `std::vector` 数据，避免 `to_array()` 的 GPU 陷阱（见 add-cuda-test 5.9）

## 8. Pre-commit 格式化

修改测试文件后必须运行：
```bash
PATH="$HOME/.local/bin:$PATH" pre-commit run --files tests/cuda/test_<module>.cpp
```
