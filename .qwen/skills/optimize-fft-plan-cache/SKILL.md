---
name: optimize-fft-plan-cache
description: Replace single-plan FFT cache with multi-plan cache to avoid ~5ms cuFFT plan recreation when alternating between FFT sizes
source: auto-skill
extracted_at: '2026-06-06T21:02:49.144Z'
---

# FFT Plan Cache 优化

## 问题

雷达 demo 交替调用 range FFT（2048 点）和 Doppler FFT（32 点），单 plan 缓存每次切换大小都 destroy + recreate，cuFFT plan 创建 ~5ms，导致 GPU FFT 和 CPU 差不多快。

## 解决方案

### cuFFT：`unordered_map` 多 plan 缓存

文件：`backends/cuda/kernels/fft/common.cuh`

- Key = `(n, batch, kind, is_f32)` — **direction 不参与 key**，因为 C2C plan 同时支持 forward/inverse，R2C/C2R 方向隐含在 kind 中
- 用 `std::unordered_map<CuFFTKey, cufftHandle, CuFFTKeyHash>`
- 最大 16 个 plan，满时 evict 最早的一个（`plans.begin()`）
- 析构时自动 `cufftDestroy` 所有 plan
- `cufft_ensure_plan()` 简化为一行：`return cufft_get_cache().get_or_create(n, batch, kind, is_f32);`

### FFTW：固定大小数组多 plan 缓存

文件：`backends/cpu/kernels/fft/common.h` + `common_impl.cpp`

- C2C：仍每次用实际 buffer 创建 plan（FFTW_ESTIMATE buffer 依赖 bug，n>=64 时用不同 buffer 执行结果错误）
- R2C/C2R：`fft_cache_find_or_create()` 用 `FFTWCacheEntry[8]` 数组，key = `(n, batch, direction, kind, is_f64)`
- 满时 evict 最早 entry：`memmove` 前移

## 关键设计决策

| 决策 | 原因 |
|------|------|
| direction 不在 cuFFT key 中 | C2C plan 同时 forward/inverse；R2C/C2R 方向隐含 |
| cuFFT 用 unordered_map | 查找 O(1)，最多 16 个 plan |
| FFTW 用固定数组 | 避免依赖 C++ STL 在 C 代码中 |
| FFTW C2C 不缓存 | FFTW_ESTIMATE 的 C2C plan 依赖创建时的 buffer 地址 |
| FFTW R2C/C2R 缓存 | `fftw_execute_dft_r2c/c2r` 不受 buffer 影响 |

## 验证方法

1. 创建两个不同大小的 complex 数组，交替调用 `ins.fft`
2. 第 1、2 次调用慢（创建 plan），后续应稳定在 ~0.1ms
3. 全量测试：`./insight_tests_cpu` 和 `./insight_tests_cuda` 全部通过

## 性能结果

- GPU [32,2048] C64 FFT: 0.026ms vs CPU 0.33ms（**12.7x** 加速比）
- 交替调用 50 次后稳定在 ~0.14ms/call（旧方案每 ~5ms）
