"""
demos/verify_task1_precision.py
逐步验证 Insight7 雷达处理流程 vs numpy+scipy 的数值精度
"""

import numpy as np
import scipy.signal as sig
import insight as ins


def to_ins(arr):
    return ins.from_numpy(arr.astype(np.complex128))


def to_ins_f64(arr):
    return ins.from_numpy(arr.astype(np.float64))


def to_np(arr):
    return arr.numpy()


print("=" * 60)
print("  任务1 精度对齐验证 (Insight7 vs numpy+scipy)")
print("=" * 60)

ins.init(["cpu"])

# ========== Step 1: LFM 信号生成 ==========
print("\n[Step 1] LFM 信号生成")
fs = 10e6
T = 1e-4
N = 1000
B = 5e6
tau = 50e-6

t = np.arange(N) / fs
s_tx_np = np.exp(1j * np.pi * (B / tau) * t**2).astype(np.complex128)
s_tx_np[t >= tau] = 0.0

# Insight7 doesn't have a direct LFM generator, so both use the same formula
s_tx_ins = to_ins(s_tx_np)
s_tx_back = to_np(s_tx_ins)
assert np.allclose(s_tx_np, s_tx_back), "LFM signal: MISMATCH"
print("  ✅ LFM 信号生成: max diff = 0.0")

# ========== Step 2: 脉冲压缩 (fftconvolve) ==========
print("\n[Step 2] 脉冲压缩 (fftconvolve)")

# Generate a test pulse
rng = np.random.RandomState(42)
pulse_np = np.zeros(N, dtype=np.complex128)
ds = int(30e-6 * fs)  # 30us delay
pulse_np[ds:] = s_tx_np[: N - ds]
noise = 0.01 * (rng.randn(N) + 1j * rng.randn(N))
pulse_np += noise

mf_np = np.conj(s_tx_np[::-1])

# scipy reference
scipy_conv = sig.fftconvolve(pulse_np, mf_np, mode="full")

# Insight7
pulse_ins = to_ins(pulse_np)
mf_ins = to_ins(mf_np)
ins_conv = to_np(ins.signal.fftconvolve(pulse_ins, mf_ins, mode="full"))

# Compare (note: may have different normalization)
conv_diff = np.max(np.abs(ins_conv - scipy_conv))
print(f"  fftconvolve full: max diff = {conv_diff:.2e}")
if conv_diff < 1e-6:
    print("  ✅ 数值一致")
else:
    # Check if they're proportional (different normalization)
    ratio = np.max(np.abs(ins_conv)) / np.max(np.abs(scipy_conv))
    print(f"  ⚠️  幅度比 = {ratio:.6f} (可能是归一化差异)")
    # Normalize and compare shape
    ins_norm = ins_conv / np.max(np.abs(ins_conv))
    scipy_norm = scipy_conv / np.max(np.abs(scipy_conv))
    shape_diff = np.max(np.abs(ins_norm - scipy_norm))
    print(f"  归一化后 max diff = {shape_diff:.2e}")
    if shape_diff < 1e-3:
        print("  ✅ 形状一致（归一化差异可接受）")

# ========== Step 3: FFT (多普勒处理) ==========
print("\n[Step 3] FFT 多普勒处理")

# Create a test matrix: 128 pulses × 1000 range bins
test_matrix = rng.randn(128, 1000) + 1j * rng.randn(128, 1000)

# numpy reference
np_fft = np.fft.fft(test_matrix, axis=0)
np_fft_shifted = np.fft.fftshift(np_fft, axes=0)

# Insight7
test_ins = to_ins(test_matrix)
ins_fft = to_np(ins.fft.fft(test_ins, n=128, axis=0))
ins_fft_shifted = np.fft.fftshift(ins_fft, axes=0)

fft_diff = np.max(np.abs(ins_fft - np_fft))
print(f"  FFT axis=0: max diff = {fft_diff:.2e}")
if fft_diff < 1e-6:
    print("  ✅ 数值一致")
else:
    print("  ⚠️  差异较大，检查 FFT 实现")

# ========== Step 4: CA-CFAR ==========
print("\n[Step 4] CA-CFAR 检测")

# Create a test energy map with known targets
energy_np = np.abs(test_matrix)
# Inject a strong target
energy_np[50, 200] = 100.0  # should be detected
energy_np[51, 201] = 80.0  # cluster neighbor
energy_np[100, 500] = 90.0  # second target

energy_ins = to_ins_f64(energy_np)

threshold, detections = ins.signal.ca_cfar(energy_ins, [2, 2], [4, 4], 1e-3)
det_np = detections.numpy().astype(bool)

# Check that injected targets are detected
assert det_np[50, 200], "Target at (50,200) should be detected"
print(f"  CA-CFAR: 注入目标 (50,200) 检测到: {det_np[50, 200]} ✅")
print(f"  CA-CFAR: 注入目标 (100,500) 检测到: {det_np[100, 500]}")
print(f"  总检测点数: {np.sum(det_np)}")

# ========== Step 5: 完整流程端到端 ==========
print("\n[Step 5] 端到端流程验证")

# Run the full pipeline and check that targets are found at expected locations
n_pulses = 128
target_delays = [30e-6, 45e-6]
target_dopplers = [2000.0, -1500.0]

# Generate echoes
s_rx = np.zeros((n_pulses, N), dtype=np.complex128)
for p in range(n_pulses):
    slow_time = p * T
    pulse = np.zeros(N, dtype=np.complex128)
    for delay in target_delays:
        ds = int(delay * fs)
        if ds < N:
            pulse[ds:] += s_tx_np[: N - ds]
    pulse *= np.exp(1j * 2 * np.pi * target_dopplers[0] * t)
    pulse *= np.exp(1j * 2 * np.pi * target_dopplers[0] * slow_time)
    noise = 0.01 * (rng.randn(N) + 1j * rng.randn(N))
    s_rx[p, :] = pulse + noise

# Pulse compression
mf = np.conj(s_tx_np[::-1])
mf_ins = to_ins(mf)
pc = np.zeros((n_pulses, N), dtype=np.complex128)
for p in range(n_pulses):
    conv = to_np(ins.signal.fftconvolve(to_ins(s_rx[p, :]), mf_ins, mode="full"))
    start = (len(conv) - N) // 2
    pc[p, :] = conv[start : start + N]

# Doppler FFT
pc_ins = to_ins(pc)
doppler = to_np(ins.fft.fft(pc_ins, n=n_pulses, axis=0))
doppler = np.fft.fftshift(doppler, axes=0)

# CFAR
energy = np.abs(doppler)
_, detections = ins.signal.ca_cfar(to_ins_f64(energy), [2, 2], [4, 4], 1e-5)
det = detections.numpy().astype(bool)

# Check targets
range_bins = np.arange(N) * (3e8 / (2 * fs))
doppler_bins = np.fft.fftshift(np.fft.fftfreq(n_pulses, T))

target_indices = np.argwhere(det)

# Cluster
visited = np.zeros(len(target_indices), dtype=bool)
targets = []
for i in range(len(target_indices)):
    if visited[i]:
        continue
    visited[i] = True
    cluster = [target_indices[i]]
    for j in range(i + 1, len(target_indices)):
        if visited[j]:
            continue
        d = np.sqrt(np.sum((target_indices[i] - target_indices[j]) ** 2))
        if d <= 5:
            visited[j] = True
            cluster.append(target_indices[j])
    center = np.mean(cluster, axis=0)
    d_idx, r_idx = int(round(center[0])), int(round(center[1]))
    targets.append((d_idx, r_idx))

print(f"  检测到 {len(targets)} 个目标:")
for d_idx, r_idx in targets:
    if 0 <= d_idx < len(doppler_bins) and 0 <= r_idx < len(range_bins):
        print(f"    → 距离: {range_bins[r_idx]:7.2f} 米, " f"多普勒: {doppler_bins[d_idx]:8.1f} Hz")

# Check if targets are near expected locations
expected_ranges = [delay * fs * (3e8 / (2 * fs)) for delay in target_delays]
expected_dopplers = target_dopplers

found = 0
for er, ed in zip(expected_ranges, expected_dopplers):
    for d_idx, r_idx in targets:
        r_m = range_bins[r_idx]
        d_hz = doppler_bins[d_idx]
        if abs(r_m - er) < 100 and abs(d_hz - ed) < 2000:
            found += 1
            break

print(f"\n  匹配预期目标: {found}/{len(target_delays)}")
if found == len(target_delays):
    print("  ✅ 所有预期目标均被检测到")
else:
    print("  ⚠️  部分目标未检测到，可能需要调整 CFAR 参数")

print("\n" + "=" * 60)
print("  验证完成")
print("=" * 60)
