---
name: fix-batched-matmul-3d
description: Fix batched matmul for 3D+ arrays — only broadcast batch dimensions, not matrix dimensions
source: auto-skill
extracted_at: '2026-05-31T03:00:00.000Z'
---

# Fix Batched Matmul for 3D+ Arrays

## Problem

The `matmul` function's batched path (Case 5 in `src/ops/linalg.cpp`) incorrectly used `broadcast_shape()` on the **entire** shape including the last two matrix dimensions. For 3D arrays like `[1, 1, 2] @ [1, 2, 2]`, this caused:

1. **Wrong output shape**: `broadcast_shape([1,1,2], [1,2,2])` = `[1,2,2]` instead of correct `[1,1,2]`
2. **Wrong matrix dimensions**: The output m×n should come from `a.dim(-2)` and `b.dim(-1)`, NOT from broadcasting

## Root Cause

```cpp
// ❌ WRONG — broadcasts ALL dimensions including matrix dims
Shape bc_shape = broadcast_shape(a.shape(), b.shape());
int64_t m = bc_shape.dim(ndim - 2);  // Wrong! Gets broadcast result
int64_t n = bc_shape.dim(ndim - 1);  // Wrong! Gets broadcast result
out_shape = bc_shape;                 // Wrong! Includes matrix dims
```

## Fix

Only broadcast the **batch dimensions** (all dimensions except the last 2), and derive m,n from the input matrices directly:

```cpp
// ✅ CORRECT — only broadcast batch dimensions
int64_t m = a.shape().dim(ndim_a - 2);   // From a's row dim
int64_t k_a = a.shape().dim(ndim_a - 1); // From a's col dim
int64_t k_b = b.shape().dim(ndim_b - 2); // From b's row dim
int64_t n = b.shape().dim(ndim_b - 1);   // From b's col dim

INS_CHECK(k_a == k_b, "matmul: shape mismatch for batch matrices");

// Broadcast batch dimensions only (exclude last 2)
int max_batch_ndim = std::max(ndim_a - 2, ndim_b - 2);
std::vector<int64_t> batch_dims(max_batch_ndim);
for (int i = 0; i < max_batch_ndim; ++i) {
    int64_t a_dim = (i < ndim_a - 2) ? a.shape().dim(ndim_a - 3 - i) : 1;
    int64_t b_dim = (i < ndim_b - 2) ? b.shape().dim(ndim_b - 3 - i) : 1;
    if (a_dim != 1 && b_dim != 1 && a_dim != b_dim)
        INS_THROW("matmul: batch dimensions not broadcastable");
    batch_dims[max_batch_ndim - 1 - i] = std::max(a_dim, b_dim);
}

// Output shape = batch_dims + [m, n]
std::vector<int64_t> out_dims(batch_dims);
out_dims.push_back(m);
out_dims.push_back(n);
out_shape = Shape(out_dims);
```

Also broadcast the **input arrays** to match batch dimensions before launching the kernel:

```cpp
// Build broadcast shapes for inputs
std::vector<int64_t> a_full(batch_dims);
a_full.push_back(m);
a_full.push_back(k_a);
std::vector<int64_t> b_full(batch_dims);
b_full.push_back(k_b);
b_full.push_back(n);

if (need_bc) {
    a_work = broadcast_to(a_work, Shape(a_full));
    b_work = broadcast_to(b_work, Shape(b_full));
}
```

## Verification

Test with KalmanFilter predict/update which uses 3D batched matmul:

```cpp
// [1, 1, 2] @ [1, 2, 1] should give [1, 1, 1], NOT [1, 2, 2]
Array H = zeros(Shape({1, 1, 2}), DType::F64, CPUPlace());
Array x = zeros(Shape({1, 2, 1}), DType::F64, CPUPlace());
Array result = matmul(H, x);
assert(result.shape() == Shape({1, 1, 1}));  // Correct!
```

## Key Insight

`broadcast_shape()` is designed for **element-wise** operations where ALL dimensions are broadcast. For **matmul**, only the batch dimensions (leading dimensions) should be broadcast — the matrix dimensions (last 2) have their own rules: `m` from `a[-2]`, `n` from `b[-1]`, inner `k` must match.

## Where This Matters

- KalmanFilter `predict()`: `F @ P @ F^T + Q` where F, P are `[points, dim_x, dim_x]`
- KalmanFilter `update()`: `H @ P @ H^T + R`, `K = P @ H^T @ S_inv`, etc.
- Any batched linear algebra with 3D+ arrays
