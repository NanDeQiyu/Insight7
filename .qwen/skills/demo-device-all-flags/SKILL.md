---
name: demo-device-all-flags
description: Add --device all/--timer/--info CLI flags to demos across all languages for competition scoring
source: auto-skill
extracted_at: '2026-06-08T12:27:09.662Z'
---

# Demo CLI Flags: --device all / --timer / --info

Pattern for adding CPU/GPU comparison (`--device all`), timing control
(`--timer`), and device memory info (`--info`) flags to competition demos
across 4 languages (C++, Python, Lua, Julia).

## Why

Competition requirements:
1. **cudaEvent timing** (5pts) → `--timer`
2. **Pure CPU comparison program** (5pts) → `--device all`
3. **Memory leak detection** (5pts) → `--info`

## Flag Design

| Flag | Default | Behavior |
|------|---------|----------|
| `--device cpu/gpu/all` | "default" (auto-detect) | `all` = run each frame on CPU then GPU, compare outputs, print `max_diff` |
| `--timer` | off | Show per-step timing breakdown (gen/filt/spl/demod/stft/cwt/corr/peak/kalman) |
| `--info` | off | Print CPU/GPU memory info via `device_memory_info` API |

When `--device gpu` but GPU unavailable → fallback to CPU with warning.
When `--device all` but GPU unavailable → run CPU only with warning.

## Implementation Pattern (per language)

### Common Steps

1. **Extend CLI parser** to accept `"all"` as a device choice + boolean flags
2. **Add `Array energy` or `Array smoothed` to frame result** for comparison
3. **In the frame loop**, when `--device all`:
   a. Initialize cache on CPU → run frame → store result
   b. Initialize cache on GPU → run frame → store result
   c. Transfer GPU arrays to CPU → compute `max(abs(cpu - gpu))`
   d. Print `CPU: xxx ms | GPU: xxx ms | max_diff=1.2e-07`
4. **Conditional timing output**: wrap detailed step breakdown with `args.timer` check
5. **Memory info**: call `device_memory_info(0, 0)` for CPU and `(1, 0)` for GPU

### Device Resolution Logic

```
Input device arg → resolve to effective device:
  "default" → has_device(GPU) ? "gpu" : "cpu"
  "gpu" but !has_device(GPU) → warn + fallback to "cpu"
  "all" → run both, if !has_device(GPU) warn + run CPU only
```

### C++ Implementation

```cpp
// Args struct
struct Args {
  std::string device = "default";
  int seed = 42;
  int iterations = 0;
  bool timer = false;
  bool info = false;
};

// FrameResult needs an Array field for comparison
struct FrameResult {
  // ... existing fields ...
  Array energy;  // ← ADD for --device all comparison
};

// In run_frame(), populate the comparison array
result.energy = energy;  // or result.smoothed = smoothed;

// In main(), for --device all mode:
if (args.device == "all") {
  // CPU run
  init_cache(CPUPlace());
  FrameResult cpu_r = run_frame(...);

  // GPU run
  init_cache(GPUPlace(0));
  FrameResult gpu_r = run_frame(...);

  // Compare
  Array diff = abs(cpu_r.energy - gpu_r.energy.to(CPUPlace()));
  double max_diff = max(diff).item<double>();
  printf("CPU: %.2f ms | GPU: %.2f ms | max_diff=%.2e\n",
         cpu_r.total_ms, gpu_r.total_ms, max_diff);
}

// --timer: conditional detail print
if (args.timer) {
  // print detailed step breakdown
} else {
  // print simplified (total time + targets only)
}

// --info: print memory
#include "insight/c_api/memory.h"  // or #include "insight/core/place.h"
if (args.info) {
  size_t total, free;
  device_memory_info(DeviceKind::CPU, 0, &total, &free);
  printf("[Memory] CPU: total=%zuMB used=%zuMB\n",
         total/1024/1024, (total-free)/1024/1024);
  if (has_device(DeviceKind::GPU)) {
    device_memory_info(DeviceKind::GPU, 0, &total, &free);
    printf("[Memory] GPU: total=%zuMB used=%zuMB\n",
           total/1024/1024, (total-free)/1024/1024);
  }
}
```

### Python Implementation

```python
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["cpu", "gpu", "all"], default="cpu")
    p.add_argument("--timer", action="store_true")
    p.add_argument("--info", action="store_true")
    # ... other args ...
    return p.parse_args()

# In frame loop, for --device all:
for frame in range(n_frames):
    if args.device == "all":
        _init_cache("cpu")
        cpu_r = run_detection("cpu", seed, frame, n_frames)

        _init_cache("gpu")
        gpu_r = run_detection("gpu", seed, frame, n_frames)

        diff = ins.max(ins.abs(cpu_r["energy"] - gpu_r["energy"].to(ins.CPUPlace())))
        max_diff = float(diff)
        print(f"CPU: {cpu_r['total_ms']:.2f} ms | GPU: {gpu_r['total_ms']:.2f} ms | max_diff={max_diff:.2e}")

# --timer control: wrap detail with args.timer
if args.timer:
    print(f"... detailed timing breakdown ...")
else:
    print(f"... simplified output ...")

# --info:
if args.info:
    cpu_total, cpu_free = ins.device_memory_info(0, 0)  # 0=CPU, 1=GPU
    print(f"[Memory] CPU: total={cpu_total//1024//1024}MB used={(cpu_total-cpu_free)//1024//1024}MB")
    if ins.has_device("gpu"):
        gpu_total, gpu_free = ins.device_memory_info(1, 0)
        print(f"[Memory] GPU: total={gpu_total//1024//1024}MB used={(gpu_total-gpu_free)//1024//1024}MB")
```

### Lua Implementation

```lua
-- CLI args
local args = { device = "cpu", timer = false, info = false }
local i = 1
while i <= #arg do
  if arg[i] == "--device" and i < #arg then
    args.device = arg[i + 1]; i = i + 1
  elseif arg[i] == "--timer" then
    args.timer = true
  elseif arg[i] == "--info" then
    args.info = true
  end
  i = i + 1
end

-- --info with math.floor for Lua 5.1 (LuaJIT) compatibility:
-- Use math.floor(x / (1024*1024)), NOT x // (1024*1024) (Lua 5.3+ only)!
if args.info then
  local cpu_total, cpu_free = ins.device_memory_info(0, 0)
  print(string.format("[Memory] CPU: total=%dMB used=%dMB",
    math.floor(cpu_total / (1024*1024)),
    math.floor((cpu_total - cpu_free) / (1024*1024))))
end

-- For --device all, transfer GPU arrays unconditionally:
-- ❌ Avoid: if gpu_smoothed:place() ~= ins.CPUPlace() then ... end
--    (place is a userdata, not callable as a method)
-- ✅ Use: local gpu_smoothed = gpu_r.smoothed:to(ins.CPUPlace())
```

### Julia Implementation

```julia
# CLI args in main()
local device_flag = "cpu"
local timer_flag = false
local info_flag = false
local i = 1
while i <= length(ARGS)
    if ARGS[i] == "--device" && i < length(ARGS); device_flag = ARGS[i+1]; i += 1
    elseif ARGS[i] == "--timer"; timer_flag = true
    elseif ARGS[i] == "--info"; info_flag = true
    end
    i += 1
end

# device_memory_info needs Int64 arguments, NOT Int32!
if info_flag
    cpu_total, cpu_free = Insight.device_memory_info(Int64(0), Int64(0))  # ✅
    # ❌ Insight.device_memory_info(Int32(0), Int32(0))  # MethodError!
end
```

## Key Pitfalls

1. **`device_memory_info` parameter types differ per language**: Python/Lua use `int` (0=CPU, 1=GPU), Julia needs `Int64`, C++ uses `DeviceKind` enum
2. **LuaJIT (5.1) vs Lua 5.3**: No `//` operator, use `math.floor(x / y)` instead
3. **Lua `place` is userdata, not callable**: Access `arr:to(CPUPlace())` directly, don't check `arr:place()`
4. **Cache re-initialization**: For `--device all`, must call `_init_cache/cpu` and `_init_cache/gpu` separately between runs because the cache is device-specific
5. **Timing output must be conditional**: Don't always show detailed breakdown; wrap with `--timer` check
6. **GPU unavailable fallback**: Both `--device gpu` and `--device all` must handle missing GPU gracefully
