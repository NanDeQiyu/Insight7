---
name: fix-cuda-kernel-view-offset
description: "CUDA kernels that iterate over input array elements must add src->offset when computing source byte address, or the kernel reads from byte 0 instead of the view's start."
source: auto-skill
extracted_at: '2026-06-06T20:24:00.000Z'
---

# Fix CUDA Kernel View Offset Access

## The Problem

When a CUDA kernel reads input elements using a flat linear index (e.g. iterating `linear` from 0 to `src->numel`), it must account for `src->offset` to handle **views** (sliced arrays). Without the offset, the kernel reads from byte 0 of the parent buffer instead of the view's start.

**Symptoms:**
- `fftshift` on GPU returns identity (no swapping), because the concat kernel reads from byte 0 for both halves
- User sees "rectangle" pattern when plotting: red crosses at bottom-left/top-right but energy peaks at top-left/bottom-right
- Any operation using `slice` + `concat` (fftshift, ifftshift) produces wrong output on GPU
- CPU backend works correctly; GPU backend gives wrong results

**Root cause:**
```cpp
// ❌ BUG: ignores src->offset — always reads from byte 0 of parent buffer
cudaMemcpy(dst, static_cast<const char*>(src->data) + linear * element_size, ...);
```

The `InsightArray` struct has field `int64_t offset` at struct offset `0xC0` (after `data`, `dims`, `strides`, `ndim`, `numel`, `dtype`, `device_type`, `device_id`). For views created by `x.slice(ax, start, end, step)`:
- `src->data` = parent buffer start (same as parent)
- `src->offset` = element offset from parent buffer start

## The Fix

Add `src->offset` to the source byte address calculation:

```cpp
// ✅ FIX: account for view offset
int64_t src_offset = src->offset + linear;
cudaMemcpy(dst, static_cast<const char*>(src->data) + src_offset * element_size, ...);
```

## Where to check

Search CUDA kernels that iterate over `src->numel` elements using `cudaMemcpy` or direct pointer arithmetic:

```bash
grep -r "cudaMemcpy.*src->data.*linear" backends/cuda/kernels/
grep -r "linear \* element_size" backends/cuda/kernels/
```

Each occurrence that reads source data using a flat linear index should add `src->offset`.

## Frontend workaround (when kernel fix is blocked)

If the kernel fix doesn't take effect (e.g., .so caching, build issues), apply a frontend workaround in the C++ API:

```cpp
// In the frontend function (e.g., src/ops/fft.cpp: fftshift):
auto last = x.slice(ax, mid, n, 1);
auto first = x.slice(ax, 0, mid, 1);

// Make contiguous copies to avoid view offset issues in backend kernels
Array last_contig = contiguous(last);
Array first_contig = contiguous(first);
Array result = concat({last_contig, first_contig}, ax);
```

This creates temporary contiguous copies that have `offset=0`, bypassing the kernel bug. Apply to both `fftshift` and `ifftshift`.

## Testing

After fixing, verify in all 4 languages (C++, Python, Lua, Julia) for both CPU and GPU:

| Test | Input | Expected output |
|------|-------|-----------------|
| fftshift 1D | `[1..8]` | `[5,6,7,8,1,2,3,4]` |
| fftshift fftfreq | `fftfreq(32, 100e-6)` | matches `np.fft.fftshift(np.fft.fftfreq(32, 100e-6))` |
| fftshift 2D | `arange(24).reshape(4,6)` | matches `np.fft.fftshift(...)` |
| fftshift axis | same 2D, axis=1 | matches `np.fft.fftshift(..., axes=1)` |
| ifftshift 1D | `[1..8]` | `[5,6,7,8,1,2,3,4]` (same as fftshift for even N) |
| ifftshift roundtrip | random array | `fftshift(ifftshift(x)) == x` |

C++ test pattern (GPU):
```cpp
TEST_F(FFTTestGPU, FFTShift) {
  set_device(GPUPlace(0));
  int n = 8;
  Array x = arange(0.0, static_cast<double>(n), 1.0, DType::F64);
  Array y = fft::fftshift(x, 0);
  Array y_cpu = y.to(CPUPlace());
  EXPECT_NEAR(y_cpu.at(0).item<double>(), 4.0, 1e-10);
  EXPECT_NEAR(y_cpu.at(4).item<double>(), 0.0, 1e-10);
}
```

Python test pattern:
```python
def test_fftshift_fftfreq_gpu(self):
    x = ins.fftfreq(32, 100e-6).to(GPU)
    result = ins.fftshift(x)
    assert_allclose(to_numpy(result), np.fft.fftshift(np.fft.fftfreq(32, 100e-6)), atol=1e-10)
```

## Debugging workflow

1. **Reproduce**: Run `ins.fftshift(ins.fftfreq(N, d))` on GPU and compare with NumPy
2. **Isolate**: Test `concat([x[16:32], x[0:16]], 0)` directly (not through fftshift)
3. **Compare CPU vs GPU**: Both use same C++ frontend code; only the backend kernel differs
4. **Check binary**: If kernel fix doesn't take effect, verify `.so` is loaded:
   ```
   ldd /proc/$$/exe | grep cuda_backend
   nm -C libinsight_cuda_backend.so | grep concat_kernel
   ```
5. **Frontend workaround**: Add `contiguous()` calls as a safe fallback
