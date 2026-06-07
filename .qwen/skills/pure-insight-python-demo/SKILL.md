---
name: pure-insight-python-demo
description: Template for pure Insight7 demos with MF FFT caching, device-aware init, --plot matplotlib rendering, and no numpy in core path
source: auto-skill
extracted_at: '2026-06-06T20:55:00.000Z'
---

# Pure Insight7 Python Demo Template

## Overview

Architecture principle: **compute script is pure Insight7; matplotlib goes in separate `_plot_*.py` modules.**

- `radar_detection.py` — imports only `argparse, math, os, time, insight` (no numpy/matplotlib)
- `_plot_radar.py` — matplotlib plotting, only loaded when `--plot` flag is used
- Pure compute mode works identically across C++/Lua/Julia; `--plot` is Python-only

## CLI Pattern (use `--iterations` and `--plot`, NOT `--images`/`--benchmark`)

```python
import argparse
import os
import time
import insight as ins

def parse_args():
    parser = argparse.ArgumentParser(description="Demo Title (Insight7)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--iterations", type=int, default=0,
                        help="纯计算轮数, 报告 FPS (0=不运行)")
    parser.add_argument("--plot", action="store_true",
                        help="启用 matplotlib 绘图 (仅 Python)")
    parser.add_argument("--output", type=str, default=".",
                        help="图片输出目录")
    return parser.parse_args()
```

## Device Routing (unchanged)

```python
def run_demo(device="cpu"):
    if device == "gpu" and ins.has_device("gpu"):
        ins.load_backend("cuda")
        ins.set_device(ins.GPUPlace(0))
    else:
        if device == "gpu":
            print("  [警告] GPU 不可用，回退到 CPU")
        ins.set_device(ins.CPUPlace())
        device = "cpu"
    # ... demo logic ...
    return {"device": device, ...}
```

## Plot Module Pattern (`--plot` lazy import)

In the main script — plot is triggered by `--plot` and dynamically imports:

```python
if args.plot:
    from _plot_demo import save_plots   # <-- this file may import numpy/matplotlib
    save_plots(result, args.output)
```

The separate `_plot_demo.py` file:

```python
"""
_plot_demo.py
Plotting module for demo. Only imported when --plot is used.
Allowed to import numpy and matplotlib.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_plots(r, output_dir="."):
    """Generate visualization figure(s)."""
    # r["some_array"].numpy() to get np arrays from Insight7 results
    data = r["some_array"].numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # ... matplotlib commands ...
    path = os.path.join(output_dir, f"demo_{r['device']}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")
```

## Main Entry

```python
if __name__ == "__main__":
    args = parse_args()
    ins.init()
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("  Demo Title (Insight7)")
    print("=" * 60)

    result = run_demo(args.device)
    print_result(result)  # text output only, no plot

    if args.plot:
        from _plot_demo import save_plots
        save_plots(result, args.output)

    # FPS benchmark
    if args.iterations > 0:
        print(f"\n[基准测试] {args.iterations} 轮纯计算...")
        times = []
        for i in range(args.iterations):
            t0 = time.time()
            run_demo(args.device)
            times.append(time.time() - t0)
        avg_ms = sum(times) / len(times) * 1000
        fps = 1.0 / (sum(times) / len(times))
        print(f"  平均: {avg_ms:.2f} ms/轮  FPS: {fps:.2f}")

    print("\n完成！")
```

## Device-Aware Cache Initialization

For demos that use pre-computed data (matched filter FFT, LFM signal, etc.), initialize a **global cache** with the correct device before the frame loop. This avoids per-frame recomputation and ensures GPU arrays stay on GPU:

```python
_S_TX = None           # LFM 发射信号 (或其它共享数据)
_MF_FFT = None          # 预计算的 FFT
_N_FFT = 1

def _init_cache(device="cpu"):
    """一次性初始化，创建在正确的设备上。"""
    global _S_TX, _MF_FFT, _N_FFT

    # 关键：确保数组创建在正确的设备上
    if device == "gpu" and ins.has_device("gpu"):
        ins.load_backend("cuda")
        ins.set_device(ins.GPUPlace(0))

    # 创建共享数据（LFM 信号、匹配滤波器 FFT 等）
    _S_TX = ins.to_complex(...)  # 自动在 GPU 上创建
    _MF_FFT = ins.fft(..., n, 0)
```

In `run_demo()`, use the cached global instead of recomputing:

```python
def run_demo(device="cpu", ...):
    # 设备选择（与 _init_cache 一致）
    ...
    # 使用缓存
    pc = _my_processing(_S_TX, _MF_FFT, ...)
```

In `main`, call `_init_cache` before the frame loop:

```python
if __name__ == "__main__":
    args = parse_args()
    ins.init()
    _init_cache(args.device)  # 一次性初始化
    ...
    for frame in range(n_frames):
        result = run_demo(args.device, frame, n_frames)
```

## GPU Batched FFT on axis=0

Insight7's `ins.fft(x, n, axis)` supports batched FFT along any axis. This has been verified correct on GPU (max diff vs numpy: 1.82e-12 for complex128).

**Usage:**
```python
# FFT along axis 0 (1000 independent 32-point FFTs on GPU)
spec = ins.fft(data, N_PULSES, 0) / float(N_PULSES)
spec = ins.fftshift(spec, 0)
```

**Performance note:** For small FFT sizes (<64 points), cuFFT batched FFT on GPU may be slower than CPU numpy+MKL. If performance is critical, benchmark both paths. For larger sizes (≥256 points), GPU FFT is faster.

## Scalar Extraction Without numpy Import

The main script must NOT `import numpy`. Use Insight7 Array methods:

```python
# ✅ Allowed — uses Insight7 API bridge, not a script-level numpy import
val = ins.mean(arr).numpy().item()
val = arr.at([i, j]).numpy().item()
val = arr.numpy().item()

# Shape extraction — shape() is a method
n = arr.shape()[0]
# For multi-dim shapes, don't call list() on Shape — use index access
s = arr.shape()
rows = int(s[0])
cols = int(s[1])
```

## Non-zero Finding (Pure Python, No numpy)

`ins.Array` has no `.nonzero()` method. Use Python iteration:

```python
def _find_nonzero(det):
    """Return list of (row, col) tuples where det is True."""
    rows = int(det.shape()[0])
    cols = int(det.shape()[1])
    points = []
    for i in range(rows):
        for j in range(cols):
            if det.at([i, j]).numpy().item():
                points.append((i, j))
    return points
```

## Complex Number Handling

```python
# abs() on complex may return complex128 — compute magnitude manually
mag = ins.sqrt(ins.real(arr) ** 2 + ins.imag(arr) ** 2)

# Scalar complex for multiplication
rot = ins.to_complex(ins.full([1], cos_val, ins.float64),
                     ins.full([1], sin_val, ins.float64))
result = arr * rot
```

## Performance Timing

```python
timings = {}
t0 = time.time()
# ... step ...
timings["step_name"] = time.time() - t0
timings["total"] = sum(timings.values())
```

## Existing Demos Using This Pattern

- `demos/python/radar_detection.py` + `_plot_radar.py` — Radar detection with CA-CFAR
- `demos/python/signal_pipeline.py` + `_plot_pipeline.py` — Multi-domain feature extraction
