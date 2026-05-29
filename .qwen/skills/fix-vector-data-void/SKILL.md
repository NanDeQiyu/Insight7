---
name: fix-vector-data-void
description: Workaround for nvcc errors involving std::vector::data() returning void in some contexts.
source: auto-skill
extracted_at: '2026-05-29T07:30:49.130Z'
---

# Fix `std::vector::data()` `void` Error

**Error**: `a value of type "void" cannot be used to initialize an entity of type "void *"` when using `vec.data()`.

**Context**: This happens in some CUDA toolkit versions (like 11.x) when passing `std::vector::data()` directly to functions expecting `void*`, especially `cudaMemcpy`.

**Fix**:
Explicitly cast the return value or assign it to a `void*` variable first.

```cpp
std::vector<float> h_data(100);
// Bad (sometimes fails)
cudaMemcpy(d_ptr, h_data.data(), size, cudaMemcpyHostToDevice);

// Good
void* ptr = h_data.data();
cudaMemcpy(d_ptr, ptr, size, cudaMemcpyHostToDevice);
```

Alternatively, use raw pointers `new float[]` / `delete[]` if `std::vector` proves too problematic in CUDA kernels.

**Note**: Sometimes the `void*` assignment trick doesn't work either (the CUDA compiler still sees `void`). In that case, **always use raw pointers** — this is the most reliable solution:

```cpp
// Most reliable: raw pointers
float *h_data = new float[100];
cudaMemcpy(d_ptr, h_data, size, cudaMemcpyHostToDevice);
// ... use data ...
delete[] h_data;
```

This pattern was confirmed to work consistently across all CUDA toolkit versions in this project. Prefer it over `std::vector` in all CUDA kernel files.
