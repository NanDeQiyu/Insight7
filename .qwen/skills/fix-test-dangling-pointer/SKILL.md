---
name: fix-test-dangling-pointer
description: Fix dangling pointer bug in CUDA tests where .to(CPUPlace()).data<T>() creates a temporary that is immediately destroyed.
source: auto-skill
extracted_at: '2026-05-29T13:30:07.492Z'
---

# Fix Test Dangling Pointer Bug

## The Problem

The pattern `c.to(CPUPlace()).data<T>()` creates a **dangling pointer**. The `.to(CPUPlace())` returns a temporary `Array` object. The `.data<T>()` method returns a pointer to the temporary's internal buffer. The temporary is destroyed at the end of the statement (after the semicolon), freeing the buffer. The pointer is then invalid when used in subsequent assertions.

```cpp
// BUG: dangling pointer — temporary Array destroyed after this line
const float *data = c.to(CPUPlace()).data<float>();
EXPECT_FLOAT_EQ(data[0], 1.0f);  // UNDEFINED BEHAVIOR: reading freed memory
```

## The Fix

Use a **named variable** to extend the lifetime of the CPU array:

```cpp
// CORRECT: named variable keeps the array alive
Array cpu_c = c.to(CPUPlace());
const float *data = cpu_c.data<float>();
EXPECT_FLOAT_EQ(data[0], 1.0f);  // Safe: cpu_c is still alive
```

Or use the existing helper function `expect_float_equal_gpu` which already does this correctly:

```cpp
template <typename T>
void expect_float_equal_gpu(const Array &gpu_arr,
                            const std::vector<T> &expected, T tol = 1e-5) {
  Array cpu_arr = gpu_arr.to(CPUPlace());  // Named variable!
  const T *data = cpu_arr.data<T>();
  ...
}
```

## How to Detect

Search for the pattern in test files:
```bash
grep -n "\.to(CPUPlace())\.data<" tests/cuda/*.cpp
```

Any line matching this pattern is a potential dangling pointer.

## Why This Only Affects Some Tests

- Tests using `expect_float_equal_gpu` or similar helpers work because the helper stores the result in a named variable.
- Tests using the inline `.to(CPUPlace()).data<T>()` pattern fail because the temporary is destroyed immediately.

## Example Fix (2026-05-29)

Fixed 22 tests across `test_manipulation.cpp`, `test_unary.cpp`, and `test_indexing.cpp` that all used this pattern. The garbage data values (like `1.0578383e+24`, `3.0687035e-41`) were actually uninitialized GPU memory being read through the dangling pointer.
