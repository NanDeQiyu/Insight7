---
name: add-composite-signal-op
description: Add composite signal operators using existing primitives (no dedicated backend kernels needed)
source: auto-skill
extracted_at: '2026-06-07T17:02:48.366Z'
---

# Adding Composite Signal Operators

When implementing signal operators that can be composed from existing Insight7 primitives (FFT, math, linalg), follow this streamlined workflow instead of writing dedicated backend kernels.

## When to Use This Skill

- Operator can be expressed as a combination of existing primitives (FFT, convolution, elementwise ops, etc.)
- No sequential data dependency that requires a custom kernel
- No performance-critical hot path that composite ops can't match

Examples: `istft`, `sosfilt`, `upfirdn`, `channelize_poly`, `qmf`, `lombscargle`

## Step-by-Step Workflow

### 1. Add Declaration to Header

File: `include/insight/ops/signal/<module>.h`

```cpp
/// @brief Brief description matching scipy.signal docs.
///
/// @param x Input array (1D, F32 or F64)
/// @param param Parameter description
/// @return Output array description
Array function_name(const Array &x, double param = 1.0);
```

### 2. Add Implementation to Source

File: `src/ops/signal/<module>.cpp`

```cpp
Array function_name(const Array &x, double param) {
  INS_CHECK(x.defined(), "function_name: input is undefined");
  INS_CHECK(x.shape().ndim() == 1, "function_name: x must be 1D");

  Place cpu = CPUPlace();
  Place dev = x.place();
  DType dtype = x.dtype();

  // Move to CPU for computation (composite ops work best on CPU)
  Array x_cpu = x.to(cpu).to(DType::F64);

  // Compose using existing primitives
  // ... use fft::fft, mul, add, concat, etc.

  // Move back to original device
  Array result = /* ... */;
  if (dev.kind() != DeviceKind::CPU)
    result = result.to(dev);
  return result;
}
```

### 3. Add CPU Tests

File: `tests/cpu/cpp/test_signal_<module>.cpp`

```cpp
TEST_F(Signal<Module>TestCPU, FunctionBasic) {
  std::vector<double> input = {1.0, 2.0, 3.0};
  Array x = to_array(input);

  Array result = signal::function_name(x, param);

  // Verify output shape
  EXPECT_EQ(result.numel(), expected_len);

  // Verify values
  const double *data = result.data<double>();
  EXPECT_NEAR(data[0], expected_val, 1e-10);
}

TEST_F(Signal<Module>TestCPU, FunctionInvalidInput) {
  Array bad = to_array(std::vector<double>{1, 2, 3});
  // Test error handling
  EXPECT_THROW(signal::function_name(bad, invalid_param), std::exception);
}
```

### 4. Add CUDA Tests

File: `tests/cuda/cpp/test_signal_<module>.cpp`

```cpp
TEST_F(Signal<Module>TestGPU, FunctionBasic) {
  std::vector<double> input = {1.0, 2.0, 3.0};
  Array x = to_array(input).to(GPUPlace(0));

  Array result = signal::function_name(x, param);

  // Copy to CPU for verification
  Array result_cpu = result.to(CPUPlace());
  const double *data = result_cpu.data<double>();
  EXPECT_NEAR(data[0], expected_val, 1e-10);
}
```

### 5. Add Language Bindings

**Python** (`bindings/python/insight_py.cpp`):
```cpp
sig.def("function_name", &signal::function_name, py::arg("x"), py::arg("param") = 1.0);
```

**Python wrapper** (`bindings/python/insight/signal/<module>.py`):
```python
def function_name(x, param=1.0):
    """Brief docstring."""
    return _signal.function_name(x, param)
```

**Python __init__.py**: Add to imports and `__all__`.

**Lua** (`bindings/lua/insight_lua.cpp`):
```cpp
sig["function_name"] = &ins::signal::function_name;
// OR for optional params:
sig["function_name"] = [](const ins::Array &x, sol::optional<double> param) {
  return ins::signal::function_name(x, param.value_or(1.0));
};
```

**Lua wrapper** (`bindings/lua/insight/signal/<module>.lua`):
```lua
M.function_name = _wrap({ "x", "param" }, function(x, param)
  return sig.function_name(x, param or 1.0)
end)
```

**Julia C ABI** (`bindings/julia/insight_julia_capi.cpp`):
```cpp
Array *insight_jl_function_name(const Array *x, double param) {
  static thread_local Array result;
  result = signal::function_name(*x, param);
  return &result;
}
```

**Julia wrapper** (`bindings/julia/src/modules/signal/<module>.jl`):
```julia
function function_name(x::InsightArray, param::Float64 = 1.0)
    ptr = ccall((:insight_jl_function_name, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), x, param)
    if ptr == C_NULL
        error("function_name failed")
    end
    return InsightArray(ptr)
end
```

**Julia exports**: Add to `Insight.jl` export list and `const` alias section.

### 6. Build and Test

```bash
cd build && make -j24
# Run new tests
./tests/insight_tests_cpu --gtest_filter="*<NewTests>*"
cd tests && ./insight_tests_cuda --gtest_filter="*<NewTests>*"
# Run full suite
./tests/insight_tests_cpu
./tests/insight_tests_cuda
# Pre-commit
cd .. && PATH="$HOME/.local/bin:$PATH" pre-commit run --all-files
```

## Common Patterns

### Sequential Filtering (sosfilt)
```cpp
// Direct Form II transposed for each biquad section
for (int64_t s = 0; s < n_sections; ++s) {
  double w1 = 0, w2 = 0;
  for (int64_t i = 0; i < n; ++i) {
    double wi = y[i] - a1*w1 - a2*w2;
    temp[i] = b0*wi + b1*w1 + b2*w2;
    w2 = w1; w1 = wi;
  }
  y.swap(temp);
}
```

### Overlap-Add Synthesis (istft)
```cpp
// IFFT each frame, apply window, overlap-add
for (int64_t i = 0; i < n_frames; ++i) {
  Array col = extract_column(Zxx, i);
  Array segment = fft::ifft(col, nfft);
  segment = real(segment) * win;
  output[start:start+nperseg] += segment;
  norm[start:start+nperseg] += win * win;
}
output /= norm;  // Normalize
```

### Polyphase Decomposition (channelize_poly)
```cpp
// Reshape filter into [n_channels x n_taps], convolve each with input column
for (int64_t k = 0; k < n_channels; ++k) {
  std::vector<double> h_k = extract_polyphase(h, k, n_channels);
  std::vector<double> x_k = extract_column(x, k, n_channels);
  // Convolve h_k with x_k
}
```

## Pitfalls

1. **`ndim()` doesn't exist on Array**: Use `shape().ndim()` instead
2. **`cast()` doesn't exist**: Use `array.to(DType::F64)` instead
3. **`Shape` uses `.dims()[i]` not `[i]`**: Access dimensions via `shape().dims()`
4. **CPU computation**: Composite ops work best on CPU; move data with `.to(CPUPlace())`
5. **DType preservation**: Convert to F64 for computation, convert back to original dtype at end
6. **Device restoration**: Move result back to original device if input was on GPU
7. **Column-major vs row-major**: Julia bindings use column-major; multi-dim arrays need care
8. **std::vector as intermediate**: Use `std::vector<double>` for CPU-only computations
