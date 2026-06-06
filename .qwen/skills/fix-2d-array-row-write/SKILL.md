---
name: fix-2d-array-row-write
description: Preallocated 2D array row assignment (pc[p] = array) silently fails for large arrays — use ins.stack instead
source: auto-skill
extracted_at: '2026-06-06T19:58:07.524Z'
---

# 2D 数组逐行赋值 bug

## 问题
对预分配的 2D complex128 数组做逐行赋值 `pc[p] = fftconvolve(...)` 或 `s_rx[p] = pulse + noise` 时，**当行数 >= 128**，所有行的值变成相同的（最后写入的那一行）。行数小时（如 4）正常。

## 症状
- 所有脉冲压缩结果相同 → Doppler FFT 后能量集中在 DC(0Hz)
- 能量图看似正常但实际没有多普勒信息
- 仅在大数组（>=128行）时出现

## 根因
`pc[p]` 返回的 view 在 Insight7 的 `__setitem__` 实现中有 bug，对于大步长的 2D 数组写入不生效。多次写入只覆盖了同一片内存。

## 解决方案
**不要**使用 `pc[p] = row_array` 模式：
```python
# ❌ 错误 — 大数组时所有行变成相同值
pc = ins.zeros([N_PULSES, N], ins.complex128)
for p in range(N_PULSES):
    pc[p] = compute_some_row()  # Bug: 只有最后一行有效
```

**使用** `ins.stack(list_of_rows, 0)` 构建：
```python
# ✅ 正确 — 适用于任意行数
rows = []
for p in range(N_PULSES):
    rows.append(compute_some_row())
pc = ins.stack(rows, 0)
```

## 注意事项
- `ins.stack` 构建的 2D 数组行是**视图**（view），不是拷贝。将视图传递给某些操作（如 `fftconvolve`）可能产生错误结果（也读到错误内存）
- 如果需要按行传递给其他函数，确保传递**新创建的数组**而非视图

## 替代方案：合并循环
如果回波模拟和脉冲压缩需要串行，将两者合并到同一循环中，避免创建中间 2D 数组：
```python
rows = []
for p in range(N_PULSES):
    s_rx = generate_pulse(...)  # 返回新 Array
    pc_row = ins.signal.fftconvolve(s_rx, mf, "same")  # s_rx 是新 Array 不是 view
    rows.append(pc_row)
pc = ins.stack(rows, 0)
```
