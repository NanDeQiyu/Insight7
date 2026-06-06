"""
demos/signal_pipeline.py
多域特征提取与信号分析流水线 (Insight7)

波形生成 → FIR 滤波 → 平滑 → 特征提取 (STFT/CWT/自相关)
         → FM 解调 → Kalman 滤波 → 峰值查找

所有计算使用 Insight7 API，numpy 仅用于数据类型转换。
"""

import argparse
import os
import time
import numpy as np  # 仅用于 from_numpy/to_numpy 桥接
import insight as ins


# ============================================================
# 1. 波形生成 (纯 Insight7)
# ============================================================
def generate_waveforms(fs, duration):
    """生成 chirp + gausspulse + sawtooth + noise 合成信号。"""
    n = int(fs * duration)
    t_np = np.arange(n) / fs
    t = ins.from_numpy(t_np.astype(np.float64))

    # Chirp (50→500 Hz, 线性调频)
    sig_chirp = ins.signal.chirp(t, 50.0, duration, 500.0, method="linear") * 0.8

    # Gausspulse (fc=200 Hz)
    center = int(0.3 * n)
    seg_len = n // 4
    t_seg = t[0:seg_len] - t[seg_len // 2]
    g = ins.signal.gausspulse(t_seg, fc=200, bw=0.5)
    sig_gauss = ins.zeros([n], ins.float64)
    clip_len = min(seg_len, n - center)
    sig_gauss[center : center + clip_len] = g[0:clip_len]

    # Sawtooth (80 Hz)
    sig_saw = ins.signal.sawtooth(t * (2 * np.pi * 80), width=0.5) * 0.4

    # Noise (Insight7 RNG)
    noise = ins.randn([n], ins.float64) * 0.15

    composite = sig_chirp + sig_gauss + sig_saw + noise

    return {
        "t": t,
        "t_np": t_np,
        "fs": fs,
        "n": n,
        "chirp": sig_chirp,
        "gausspulse": sig_gauss,
        "sawtooth": sig_saw,
        "noise": noise,
        "composite": composite,
    }


# ============================================================
# 2. 滤波器设计 (纯 Insight7)
# ============================================================
def design_filters(fs):
    """设计 FIR 带通滤波器 [40, 300] Hz。"""
    numtaps = 127
    nyq = fs / 2.0
    taps = ins.signal.firwin(numtaps, [40.0 / nyq, 300.0 / nyq], pass_zero="bandpass")
    return taps, numtaps


# ============================================================
# 3. 滤波 (纯 Insight7)
# ============================================================
def apply_filter(sig_arr, taps_arr, numtaps):
    """应用 FIR 滤波器，补偿群延迟。"""
    full = ins.signal.fftconvolve(sig_arr, taps_arr, "full")
    half = numtaps // 2
    n = sig_arr.shape[0]
    return full[half : half + n]


# ============================================================
# 4. 平滑 (高斯核卷积，纯 Insight7)
# ============================================================
def gaussian_smooth(sig_arr, smooth_factor=0.1):
    """使用高斯核 + fftconvolve 平滑信号。"""
    n = sig_arr.shape[0]
    kernel_size = max(3, int(smooth_factor * n))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 构造高斯核
    x_k = np.linspace(-3, 3, kernel_size)
    kernel_np = np.exp(-0.5 * x_k**2)
    kernel_np /= kernel_np.sum()
    kernel = ins.from_numpy(kernel_np.astype(np.float64))

    # mirror-pad (手动构造 padded 数组)
    pad = kernel_size // 2
    padded = ins.zeros([n + 2 * pad], ins.float64)
    padded[pad : pad + n] = sig_arr
    # 反射填充
    for i in range(pad):
        padded[i] = sig_arr[pad - i]
        padded[n + pad + i] = sig_arr[n - 2 - i]

    result = ins.signal.fftconvolve(padded, kernel, "valid")
    return result[0:n]


# ============================================================
# 5. 特征提取 (纯 Insight7)
# ============================================================
def extract_features(sig_arr, fs):
    """提取多域特征: FM 解调, STFT, 频谱图, CWT, 自相关。"""
    features = {}

    # FM 解调: hilbert → fm_demod (返回归一化频率，乘 fs 转 Hz)
    analytic = ins.signal.hilbert(sig_arr)
    features["fm_demod"] = ins.signal.fm_demod(analytic) * fs

    # STFT
    stft_r = ins.signal.stft(sig_arr, fs=fs, nperseg=256, noverlap=192)
    features["stft_f"] = stft_r.f
    features["stft_t"] = stft_r.t
    features["stft_Sxx"] = stft_r.Sxx

    # 频谱图
    spec_r = ins.signal.spectrogram(sig_arr, fs=fs, nperseg=256, noverlap=192)
    features["spec_f"] = spec_r.f
    features["spec_t"] = spec_r.t
    features["spec_Sxx"] = spec_r.Sxx

    # CWT (Morlet 小波)
    widths_list = list(float(w) for w in range(1, 32))
    features["cwt_widths"] = widths_list
    features["cwt_matrix"] = ins.abs(ins.signal.cwt(sig_arr, ins.signal.morlet2, widths_list))

    # 自相关
    seg_len = min(int(sig_arr.shape[0]), 2048)
    seg = sig_arr[0:seg_len]
    autocorr_full = ins.signal.correlate(seg, seg, "full")
    half = autocorr_full.shape[0] // 2
    autocorr = autocorr_full[half:]
    norm_val = autocorr[0]
    if float(norm_val) != 0:
        autocorr = autocorr / norm_val
    features["autocorr"] = autocorr

    return features


# ============================================================
# 6. 峰值查找 (纯 Insight7)
# ============================================================
def find_peaks(sig_arr, order=15):
    """使用 argrelextrema 查找极值。"""
    peaks_max = ins.signal.argrelextrema(sig_arr, comparator="greater", order=order)
    peaks_min = ins.signal.argrelextrema(sig_arr, comparator="less", order=order)
    # 返回 [Array(shape=[1, N])]，取第一个并 flatten
    max_arr = peaks_max[0].numpy().flatten().astype(int) if peaks_max else np.array([], dtype=int)
    min_arr = peaks_min[0].numpy().flatten().astype(int) if peaks_min else np.array([], dtype=int)
    return max_arr, min_arr


# ============================================================
# 7. Kalman 滤波 (Insight7 KalmanFilter 类)
# ============================================================
def estimate_parameters(inst_freq_arr, fs, peaks):
    """使用 Insight7 KalmanFilter 进行频率估计。"""
    n_peaks = len(peaks)
    if n_peaks == 0:
        return ins.from_numpy(np.array([0.0])), []

    # 提取峰值处频率值
    peak_freqs_np = np.array([float(inst_freq_arr[int(p)]) for p in peaks])

    # KalmanFilter: points=n_peaks, dim_x=1, dim_z=1
    kf = ins.signal.KalmanFilter(dim_x=1, dim_z=1, points=n_peaks, dtype=ins.float64)

    # 模型矩阵 (所有点共享相同模型)
    F = ins.zeros([n_peaks, 1, 1], ins.float64)
    H = ins.zeros([n_peaks, 1, 1], ins.float64)
    Q = ins.zeros([n_peaks, 1, 1], ins.float64)
    R = ins.zeros([n_peaks, 1, 1], ins.float64)
    for i in range(n_peaks):
        F[i, 0, 0] = 1.0
        H[i, 0, 0] = 1.0
        Q[i, 0, 0] = 1e-4
        R[i, 0, 0] = 0.5

    kf.F = F
    kf.H = H
    kf.Q = Q
    kf.R = R

    # 初始化状态
    x0 = ins.zeros([n_peaks, 1, 1], ins.float64)
    P0 = ins.zeros([n_peaks, 1, 1], ins.float64)
    for i in range(n_peaks):
        x0[i, 0, 0] = peak_freqs_np[i]
        P0[i, 0, 0] = 1.0
    kf.x = x0
    kf.P = P0

    # 单步 predict + update (批量并行)
    kf.predict()
    z = ins.zeros([n_peaks, 1, 1], ins.float64)
    for i in range(n_peaks):
        z[i, 0, 0] = peak_freqs_np[i]
    kf.update(z)

    # 提取结果
    smoothed_np = kf.x.numpy().flatten()

    params = []
    for i, p in enumerate(peaks):
        if 0 <= int(p) < int(inst_freq_arr.shape[0]):
            params.append(
                {
                    "index": int(p),
                    "time": int(p) / fs,
                    "inst_freq": float(smoothed_np[i]),
                }
            )

    return smoothed_np, params


# ============================================================
# 主流水线
# ============================================================
def run_pipeline(device="cpu"):
    if device == "gpu" and ins.has_device("gpu"):
        ins.load_backend("cuda")
        ins.set_device(ins.GPUPlace(0))
    else:
        if device == "gpu":
            print("  [警告] GPU 不可用，回退到 CPU")
        ins.set_device(ins.CPUPlace())
        device = "cpu"

    fs = 5000.0
    duration = 0.5
    timings = {}

    print("=" * 60)
    print("  多域特征提取与信号分析 (Insight7)")
    print("=" * 60)

    # --- 步骤 1: 波形生成 ---
    print("\n[1/6] 波形生成...")
    t0 = time.time()
    wavs = generate_waveforms(fs, duration)
    timings["waveform"] = time.time() - t0
    print("  生成: chirp + gausspulse + sawtooth + noise")
    print(f"  信号长度: {wavs['n']} 采样点, 时长: {duration}s")
    print(f"  耗时: {timings['waveform']:.4f} 秒")

    # --- 步骤 2: 滤波器设计 ---
    print("\n[2/6] 滤波器设计 (firwin)...")
    t0 = time.time()
    taps_arr, numtaps = design_filters(fs)
    timings["filter_design"] = time.time() - t0
    print(f"  带通滤波器: {numtaps} 阶, [40, 300] Hz")

    # --- 步骤 3: 滤波 ---
    print("\n[3/6] 滤波处理 (去趋势 + 带通)...")
    t0 = time.time()
    detrended = ins.signal.detrend(wavs["composite"])
    signal_filtered = apply_filter(detrended, taps_arr, numtaps)
    timings["filtering"] = time.time() - t0
    print(f"  耗时: {timings['filtering']:.4f} 秒")

    # --- 步骤 4: 平滑 ---
    print("\n[4/6] 高斯核平滑...")
    t0 = time.time()
    signal_smooth = gaussian_smooth(signal_filtered)
    timings["smooth"] = time.time() - t0
    print(f"  耗时: {timings['smooth']:.4f} 秒")

    # --- 步骤 5: 特征提取 ---
    print("\n[5/6] 多域特征提取...")
    t0 = time.time()
    features = extract_features(signal_smooth, fs)
    timings["features"] = time.time() - t0
    print(f"  fm_demod: {features['fm_demod'].shape[0]} 点")
    print(f"  STFT:     {features['stft_Sxx'].shape}")
    print(f"  频谱图:   {features['spec_Sxx'].shape}")
    print(f"  CWT:      {features['cwt_matrix'].shape}")
    print(f"  自相关:   {features['autocorr'].shape[0]} 延迟")
    print(f"  耗时: {timings['features']:.4f} 秒")

    # --- 步骤 6: 峰值查找 + Kalman ---
    print("\n[6/6] 峰值查找 + Kalman 滤波...")
    t0 = time.time()
    peaks_max, peaks_min = find_peaks(signal_smooth, order=15)
    print(f"  相对极值: {len(peaks_max)} 极大值, {len(peaks_min)} 极小值")

    smoothed_freq, params = estimate_parameters(features["fm_demod"], fs, peaks_max)
    print(f"  Kalman 滤波: 在 {len(params)} 个峰值处估计频率")

    if params:
        print("\n  峰值参数估计:")
        print(f"    {'索引':>8} {'时间(s)':>10} {'频率(Hz)':>15}")
        print(f"    {'-'*8} {'-'*10} {'-'*15}")
        for p in params[:10]:
            print(f"    {p['index']:8d} {p['time']:10.4f} {p['inst_freq']:15.2f}")
        if len(params) > 10:
            print(f"    ... (还有 {len(params) - 10} 个)")

    timings["peaks_kalman"] = time.time() - t0
    print(f"  耗时: {timings['peaks_kalman']:.4f} 秒")

    timings["total"] = sum(timings.values())

    return {
        "wavs": wavs,
        "signal_filtered": signal_filtered,
        "signal_smooth": signal_smooth,
        "features": features,
        "smoothed_freq": smoothed_freq,
        "peaks_max": peaks_max,
        "params": params,
        "timings": timings,
        "device": device,
    }


# ============================================================
# 绘图 (matplotlib, 仅可视化)
# ============================================================
def save_plots(result, output_dir, prefix):
    """生成 6 子图可视化。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [跳过绘图] matplotlib 不可用")
        return

    wavs = result["wavs"]
    features = result["features"]
    t = wavs["t_np"]
    fs = wavs["fs"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # (1) 原始波形
    ax = axes[0, 0]
    ax.plot(t, wavs["chirp"].numpy(), alpha=0.7, label="chirp")
    ax.plot(t, wavs["gausspulse"].numpy(), alpha=0.7, label="gausspulse")
    ax.plot(t, wavs["sawtooth"].numpy(), alpha=0.7, label="sawtooth")
    ax.set_title("Original Waveforms")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) 合成 → 滤波 → 平滑
    ax = axes[0, 1]
    ax.plot(t, wavs["composite"].numpy(), alpha=0.4, label="composite")
    ax.plot(t, result["signal_filtered"].numpy(), alpha=0.6, label="filtered")
    ax.plot(t, result["signal_smooth"].numpy(), linewidth=2, label="smoothed")
    ax.set_title("Composite → Filtered → Smoothed")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) 频谱图
    ax = axes[1, 0]
    spec_db = 10 * np.log10(features["spec_Sxx"].numpy() + 1e-12)
    ax.pcolormesh(
        features["spec_t"].numpy(),
        features["spec_f"].numpy(),
        spec_db.T,
        shading="gouraud",
        cmap="inferno",
    )
    ax.set_title("Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, fs / 2)

    # (4) CWT
    ax = axes[1, 1]
    cwt_np = features["cwt_matrix"].numpy()
    widths = features["cwt_widths"]
    ax.imshow(cwt_np, aspect="auto", cmap="viridis", extent=[t[0], t[-1], widths[-1], widths[0]])
    ax.set_title("CWT Scalogram (morlet2)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scale")

    # (5) FM 解调 + Kalman
    ax = axes[2, 0]
    fm_np = features["fm_demod"].numpy()
    smoothed_np = result["smoothed_freq"]
    ax.plot(t[: len(fm_np)], fm_np, alpha=0.4, label="FM demod (raw)")
    ax.plot(t[: len(smoothed_np)], smoothed_np, linewidth=2, label="Kalman smoothed")
    params = result["params"]
    if params:
        peak_times = [p["time"] for p in params]
        peak_freqs = [p["inst_freq"] for p in params]
        ax.scatter(peak_times, peak_freqs, c="red", s=30, zorder=5, label="Kalman est.")
    ax.set_title("FM Demodulation + Kalman Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous Freq (Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (6) 自相关
    ax = axes[2, 1]
    autocorr_np = features["autocorr"].numpy()
    lags = np.arange(len(autocorr_np)) / fs
    ax.plot(lags[:500], autocorr_np[:500])
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lag (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{prefix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  已保存: {path}")


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="多域特征提取与信号分析流水线 (Insight7)")
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="运行设备 (默认: cpu)"
    )
    parser.add_argument(
        "--images", type=int, default=0, help="生成 N 张图并报告 FPS (0=仅生成 1 张)"
    )
    parser.add_argument("--benchmark", action="store_true", help="纯计时模式 (不画图)")
    parser.add_argument("--output", type=str, default=".", help="图片输出目录 (默认: 当前目录)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ins.init()
    os.makedirs(args.output, exist_ok=True)

    result = run_pipeline(args.device)

    if not args.benchmark:
        save_plots(result, args.output, f"signal_pipeline_{args.device}")

    # --- FPS 测试 ---
    if args.images > 0:
        print(f"\n[FPS 测试] 生成 {args.images} 张图...")
        times = []
        for i in range(args.images):
            t0 = time.time()
            r = run_pipeline(args.device)
            if not args.benchmark:
                save_plots(r, args.output, f"signal_pipeline_{args.device}_frame{i}")
            times.append(time.time() - t0)
        avg = np.mean(times)
        print(f"  平均: {avg*1000:.2f} ms/帧, FPS: {1/avg:.1f}")

    # --- 性能总结 ---
    print("\n" + "=" * 60)
    print("  性能总结")
    print("=" * 60)
    for k, v in result["timings"].items():
        print(f"  {k:>20s}: {v*1000:8.2f} ms")

    if ins.has_device("gpu") and args.device == "cpu":
        print("\n[提示] GPU 可用, 使用 --device gpu 运行 GPU 版本")

    print("\n完成！")
