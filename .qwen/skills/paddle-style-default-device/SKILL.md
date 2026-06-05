---
name: paddle-style-default-device
description: Implement PaddlePaddle-style lazy GPU default device in C++ core, all languages inherit
source: auto-skill
extracted_at: '2026-06-05T14:22:08.096Z'
---

## PaddlePaddle-Style Default Device

**Problem**: PaddlePaddle defaults to GPU when available, CPU otherwise. Insight7 should match this behavior.

**Implementation** (in `src/core/place.cpp`):

```cpp
namespace {
thread_local Place g_default_device = CPUPlace();
thread_local bool g_device_explicitly_set = false;
}

Place get_device() {
  // Lazy: first call checks if GPU is available
  if (!g_device_explicitly_set && g_default_device.kind() == DeviceKind::CPU) {
    if (is_device_available(DeviceKind::GPU)) {
      g_default_device = GPUPlace(0);
    }
  }
  return g_default_device;
}

void set_device(const Place &place) {
  g_default_device = place;
  g_device_explicitly_set = true;  // User's choice is honored
}
```

**Key design**:
- `g_device_explicitly_set` flag ensures `set_device(CPUPlace())` is honored
- `is_device_available(DeviceKind::GPU)` handles ALL backends (cuda/npu/rocm/...)
- No hardcoded backend names

**Binding requirements**:
- All languages MUST expose `set_device` and `get_device`
- Python `creation.py` MUST call `get_device()` at call time, NOT at module load time:
  ```python
  # ❌ WRONG — evaluated once at import
  m.def("zeros", &zeros, py::arg("place") = get_device());
  # ✅ CORRECT — evaluated at each call
  def zeros(shape, dtype, place=None):
      if place is None:
          return _native_zeros(shape, dtype, get_device())
  ```
- CPU test suites need `set_device(CPUPlace())` in setup (conftest.py for Python, setup block for Lua/busted)

**Why:** PaddlePaddle does this in `_current_expected_place_()` — checks `is_compiled_with_cuda()` + `get_cuda_device_count()`. Insight7's HAL abstraction makes it simpler: one `is_device_available()` check.
