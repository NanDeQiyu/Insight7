---
name: numpy-partial-indexing
description: Implement NumPy-style partial indexing in C++ core — a[1] on multi-dim returns slice view, all languages benefit
source: auto-skill
extracted_at: '2026-06-05T14:54:32.756Z'
---

# NumPy-style Partial Indexing (C++ Core)

## Problem

`a[1]` on a 2D array should return a row (like NumPy), not throw "index count
mismatch". Fix should be in C++ core so all language bindings benefit.

## NumPy behavior

- `a[0]` on (3,4) → row 0, shape (4,)
- `a[0,1]` → scalar
- `a[0]` on (2,3,4) → shape (3,4)
- `a.at({1,2})` on (2,3,4) → shape (4,)

## C++ core changes

### 1. Modify `at(const std::vector<int64_t>&)` in `src/core/array.cpp`

Change the check from `indices.size() == ndim` to `indices.size() <= ndim`:

```cpp
Array Array::at(const std::vector<int64_t> &indices) const {
  INS_CHECK(defined(), "Array is not initialized");
  int ndim = shape_.ndim();
  INS_CHECK(indices.size() <= static_cast<size_t>(ndim),
            "at(): too many indices (got ", indices.size(),
            ", array has ", ndim, " dimensions)");

  // Full indexing: return scalar view (original behavior)
  if (static_cast<int>(indices.size()) == ndim) {
    int64_t elem_offset = layout_.offset;
    for (int i = 0; i < ndim; ++i) {
      int64_t idx = indices[i];
      int64_t dim_size = shape_.dim(i);
      if (idx < 0) idx += dim_size;
      INS_CHECK(idx >= 0 && idx < dim_size, "at(): index out of range");
      elem_offset += idx * strides_[i];
    }
    return Array(*this, Shape({}), Strides(), elem_offset);
  }

  // Partial indexing: indexed dims consumed, remaining dims kept as view
  int64_t new_offset = layout_.offset;
  for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
    int64_t idx = indices[i];
    int64_t dim_size = shape_.dim(i);
    if (idx < 0) idx += dim_size;
    INS_CHECK(idx >= 0 && idx < dim_size, "at(): index out of range");
    new_offset += idx * strides_[i];
  }

  std::vector<int64_t> new_dims, new_strides_vec;
  for (int i = static_cast<int>(indices.size()); i < ndim; ++i) {
    new_dims.push_back(shape_.dim(i));
    new_strides_vec.push_back(strides_[i]);
  }
  return Array(*this, Shape(new_dims), Strides(new_strides_vec), new_offset);
}
```

### 2. Add `operator[](int64_t)` in `include/insight/core/array.h`

```cpp
Array operator[](int64_t index) const;
```

Implementation in `src/core/array.cpp`:

```cpp
Array Array::operator[](int64_t index) const {
  return at({index});
}
```

### 3. `operator[](const std::string&)` — no change needed

The existing string-based operator already handles partial indexing correctly:
- When all parts are integers and `parts.size() < ndim`, it calls `at()` which
  now handles partial indexing
- When mixed int+slice, it pads with `Slice::all()` for missing trailing dims

## Language binding changes

### Python

Add four `__getitem__` overloads: int, tuple (mixed int/slice), string, py::slice.
See `add-python-slice-getitem` skill for details.

### Lua

Already works — existing `__index` handler calls `a.at({idx})` with 1-based →
0-based conversion. The C++ `at()` fix makes partial indexing work automatically.

### Julia

Add `Base.getindex(arr::InsightArray, indices::Int...)` that converts 1-based
to 0-based and calls C API `insight_jl_at_index`:

```julia
function Base.getindex(arr::InsightArray, indices::Int...)
    c_indices = [Int64(i - 1) for i in indices]
    n = Int32(length(c_indices))
    ptr = ccall((:insight_jl_at_index, LIB_INSIGHT), Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Int64}, Int32), arr, c_indices, n)
    if ptr == C_NULL
        error("Insight: indexing failed")
    end
    result = InsightArray(ptr)
    finalizer(_free, result)
    return result
end
```

C API in `insight_julia_capi.cpp`:

```cpp
Array *insight_jl_at_index(const Array *arr, const int64_t *indices,
                           int32_t n_indices) {
  try {
    std::vector<int64_t> idx(indices, indices + n_indices);
    return new Array(arr->at(idx));
  } catch (...) { return nullptr; }
}
```

## How to apply

1. Modify `at()` in `src/core/array.cpp` (indices.size() <= ndim)
2. Add `operator[](int64_t)` to header and implementation
3. Add language-specific indexing bindings
4. Add C++ tests: `a[1]` on 2D, `a[1]` on 3D, `a.at({1,2})` on 3D, 1D still scalar, too-many throws
5. Verify all existing indexing tests still pass (no regression)
