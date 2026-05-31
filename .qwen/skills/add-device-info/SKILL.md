---
name: add-device-info
description: Add device information query functions through HAL → ins:: API → language bindings
source: auto-skill
extracted_at: '2026-05-30T14:50:59.903Z'
---

# Add Device Information Functions

Pattern for exposing hardware information queries through the Insight7
plugin architecture to user-facing APIs and language bindings.

## Architecture

```
HAL (device_ext.h)  →  ins:: API (place.h/cpp)  →  Bindings (Python/Lua/Julia)
  C_DeviceInterface      free functions               module-level functions
  function pointers      namespace ins                ins.device_name("gpu")
```

## Step 1: HAL Layer (if new query needed)

If the HAL already has the function pointer (e.g., `get_compute_capability`,
`get_runtime_version`, `device_memory_stats`), skip to Step 2.

Otherwise, add a new function pointer to `C_DeviceInterface` in
`include/insight/c_api/device_ext.h`:

```cpp
C_Status (*get_device_name)(C_Device device, char *buf, size_t buf_size);
```

Place it near related functions (after `get_max_grid_dim_size`, before
Profiler section).

## Step 2: CUDA Backend Implementation

In `backends/cuda/device/cuda_device.cpp`:

```cpp
static C_Status cuda_get_device_name(C_Device device, char *buf,
                                     size_t buf_size) {
  if (!buf || buf_size == 0 || !device) return C_FAILED;
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  std::strncpy(buf, prop.name, buf_size - 1);
  buf[buf_size - 1] = '\0';
  return C_SUCCESS;
}
```

Register in the iface setup function:
```cpp
iface->get_device_name = cuda_get_device_name;
```

CPU backend stubs are not needed — the `ins::` layer handles
missing function pointers gracefully.

## Step 3: ins:: Upper-Level API

In `include/insight/core/place.h` — declare free functions:

```cpp
std::string device_name(DeviceKind kind, int device_id = 0);
int cuda_version();
int driver_version();
int compute_capability(int device_id = 0);

struct DeviceMemoryInfo { size_t total; size_t free; };
DeviceMemoryInfo device_memory(int device_id = 0);
```

In `src/core/place.cpp` — implement using HAL:

```cpp
std::string device_name(DeviceKind kind, int device_id) {
  if (kind == DeviceKind::CPU) return "CPU";
  const C_DeviceInterface *iface = get_device_interface(kind);
  if (!iface || !iface->get_device_name) return "";
  C_Device_st dev;
  dev.id = device_id;
  char buf[256] = {};
  C_Status status = iface->get_device_name(&dev, buf, sizeof(buf));
  return (status == C_SUCCESS) ? std::string(buf) : "";
}
```

Pattern: get `C_DeviceInterface*` via `get_device_interface()`, check
function pointer is non-null, create `C_Device_st` with `dev.id = device_id`,
call HAL, return result or fallback.

## Step 4: Language Bindings

### Python (pybind11)
```cpp
m.def("device_name",
    [](const std::string &kind, int device_id) {
      DeviceKind dk = (kind == "gpu" || kind == "cuda")
                          ? DeviceKind::GPU : DeviceKind::CPU;
      return device_name(dk, device_id);
    },
    py::arg("kind") = "cpu", py::arg("device_id") = 0);
m.def("cuda_version", []() { return cuda_version(); });
m.def("device_memory", [](int id) {
    auto info = device_memory(id);
    return py::make_tuple(info.total, info.free);
}, py::arg("device_id") = 0);
```

### Lua (sol2)
```cpp
m["device_name"] = [](sol::optional<std::string> kind,
                      sol::optional<int> device_id) {
  std::string k = kind.value_or("cpu");
  DeviceKind dk = (k == "gpu" || k == "cuda") ? DeviceKind::GPU : DeviceKind::CPU;
  return device_name(dk, device_id.value_or(0));
};
```

### Julia (C ABI + wrapper)

C wrapper in `insight_julia_capi.cpp`:
```cpp
void insight_jl_device_name(int32_t device_id, char *buf, size_t buf_size) {
  std::string name = device_name(DeviceKind::GPU, device_id);
  std::strncpy(buf, name.c_str(), buf_size - 1);
  buf[buf_size - 1] = '\0';
}
```

Julia wrapper in `Insight.jl`:
```julia
function device_name(device_id::Int=0)::String
    buf = Vector{UInt8}(undef, 256)
    ccall((:insight_jl_device_name, LIB_INSIGHT), Cvoid,
          (Int32, Ptr{UInt8}, Csize_t), Int32(device_id), buf, 256)
    return String(buf[1:findfirst(==(0x00), buf)-1])
end
```

## Step 5: Test

```python
# Quick smoke test
import insight as ins
ins.load_backend("cuda")
print(ins.device_name("gpu"))       # "NVIDIA A800-SXM4-80GB"
print(ins.cuda_version())            # 11080
print(ins.compute_capability())      # 80
print(ins.device_memory())           # (total, free) in bytes
```

## Key HAL Functions Already Available

| HAL pointer | Returns | Notes |
|---|---|---|
| `get_device_count` | device count | Already wrapped as `ins::device_count()` |
| `get_compute_capability` | SM version (e.g., 80) | major*10+minor |
| `get_runtime_version` | CUDA runtime ver | major*1000+minor*10 |
| `get_driver_version` | CUDA driver ver | major*1000+minor*10 |
| `device_memory_stats` | (total, free) bytes | via `cudaMemGetInfo` |
| `get_multi_process` | SM count | optional |
| `get_max_threads_per_mp` | threads/SM | optional |
| `get_max_threads_per_block` | threads/block | optional |
| `get_max_grid_dim_size` | grid dims [3] | optional |
| `get_device_name` | GPU name string | **new** — needs HAL addition |
