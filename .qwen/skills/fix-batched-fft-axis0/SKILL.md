---
name: fix-batched-fft-axis0
description: ins.fft(arr, n, axis=0) on 2D complex128 GPU arrays produces wrong results — use numpy FFT or axis=1 workaround
source: auto-skill
extracted_at: '2026-06-06T19:58:07.524Z'
---

# Batched FFT 沿 axis=0 的 bug

## 问题
`ins.fft(pc * window_2d, N_PULSES, axis=0)` 对 2D complex128 数组沿 axis=0 做批量 FFT 时，结果全集中在 DC(0Hz)。

## 鉴别方法
对比 numpy 的 1D FFT 结果：
```python
# Insight7 FFT (有 bug)
spec = ins.fft(data, n, 0)  # ❌ 全在 0Hz

# NumPy FFT (正确)
spec_np = np.fft.fft(data.numpy(), n, axis=0)  # ✅ 正确位置
```

## 适用条件
- GPU 后端 (cuFFT)
- 2D complex128 数组
- N_PULSES <= 128 时出现（已验证 32 和 128 都有问题）
- 沿 axis=0（非最后一个维度）做批量 FFT

## 解决方案

### 方案 A：使用 numpy FFT（快速，但引入 numpy 依赖）
```python
pc_np = pc_ac.numpy()
win_2d = np.reshape(window.numpy(), [N_PULSES, 1])
spec_np = np.fft.fft(pc_np * win_2d, N_PULSES, axis=0) / float(N_PULSES)
spec_np = np.fft.fftshift(spec_np, axes=0)
doppler_shifted = ins.from_numpy(spec_np)
```

### 方案 B：转置后沿 axis=1 做 FFT（纯 Insight7，但需确保数据连续）
```python
pc_t = ins.permute(pc, [1, 0])  # (N, N_PULSES)
window_2d = ins.reshape(window, [1, N_PULSES])
spec_t = ins.fft(pc_t * window_2d, N_PULSES, 1) / float(N_PULSES)
spec = ins.permute(spec_t, [1, 0])
spec = ins.fftshift(spec, 0)
```

### 方案 C：1D FFT 逐列计算（慢，仅用于验证）
```python
n_cols = int(pc.shape()[1])
result = ins.zeros([N_PULSES, n_cols], ins.complex128)
for c in range(n_cols):
    col = pc[:, c]  # 或使用字符串索引
    result[:, c] = ins.fft(col * window, N_PULSES, 0)
```

## 注意事项
- `ins.fft` 沿 axis=1（最后一个维度）工作正常
- 1D FFT `ins.fft(1d_array, n, 0)` 工作正常
- `ins.fft2d` 未测试
- 该 bug 在 CPU 后端也可能存在（未验证）
