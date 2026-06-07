"""
_plot_feature.py
多域特征提取 matplotlib 绘图模块。仅被 feature_extraction.py 的 --plot 分支导入。
允许使用 numpy 和 matplotlib。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_feature_plot(r, output_dir, frame_idx=None):
    """六子图：波形、滤波、STFT、CWT、FM解调+卡尔曼、自相关。"""
    composite = r["composite"].numpy()
    filtered = r["filtered"].numpy()
    smoothed = r["smoothed"].numpy()
    inst_freq = r["inst_freq"].numpy()
    smoothed_freq = np.array(r["smoothed_freq"])
    autocorr_list = r["autocorr_list"]
    cwt_matrix = np.abs(r["cwt_matrix"].numpy())
    params = r["params"]
    fs = 5000.0
    n = len(composite)
    t = np.arange(n) / fs

    suffix = f"_frame{frame_idx:04d}" if frame_idx is not None else ""
    title_prefix = f"Frame {frame_idx}" if frame_idx is not None else ""

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # (1) 原始波形
    ax = axes[0, 0]
    ax.plot(t, composite, alpha=0.5, label="composite")
    ax.plot(t, filtered, alpha=0.7, label="filtered")
    ax.plot(t, smoothed, linewidth=2, label="smoothed")
    ax.set_title(f"{title_prefix} Composite → Filtered → Smoothed")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2) STFT 幅度
    ax = axes[0, 1]
    Zxx = r["Zxx"].numpy()
    f_stft = r["f_stft"].numpy()
    t_stft = r["t_stft"].numpy()
    Zxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    # Zxx shape: (freq, time), pcolormesh needs (time, freq) or matching grid
    ax.pcolormesh(t_stft, f_stft, Zxx_db.T, shading="auto", cmap="magma")
    ax.set_title("STFT Magnitude (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # (3) CWT 尺度图
    ax = axes[1, 0]
    n_scales = cwt_matrix.shape[0]
    widths = np.arange(1, 1 + n_scales * 4, 4)[:n_scales]
    ax.imshow(
        cwt_matrix, aspect="auto", cmap="viridis", extent=[t[0], t[-1], widths[-1], widths[0]]
    )
    ax.set_title("CWT Scalogram (morlet2)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scale")

    # (4) FM 解调 + 卡尔曼
    ax = axes[1, 1]
    ax.plot(t, inst_freq, alpha=0.4, label="FM demod (raw)")
    ax.plot(t, smoothed_freq, linewidth=2, label="Kalman smoothed")
    if params:
        peak_times = [p[1] for p in params]
        peak_freqs = [p[2] for p in params]
        ax.scatter(peak_times, peak_freqs, c="red", s=30, zorder=5, label="peaks")
    ax.set_title("FM Demodulation + Kalman Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous Freq (Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (5) 自相关
    ax = axes[2, 0]
    autocorr = np.array(autocorr_list)
    lags = np.arange(len(autocorr)) / fs
    ax.plot(lags[:500], autocorr[:500])
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lag (s)")
    ax.grid(True, alpha=0.3)

    # (6) 峰值分布
    ax = axes[2, 1]
    peaks_max = r["peaks_max"]
    peaks_min = r["peaks_min"]
    ax.plot(t, smoothed, linewidth=1, label="smoothed signal")
    if peaks_max:
        peak_t = [p / fs for p in peaks_max]
        peak_v = [smoothed[p] for p in peaks_max]
        ax.scatter(peak_t, peak_v, c="red", s=20, zorder=5, label=f"max ({len(peaks_max)})")
    if peaks_min:
        peak_t = [p / fs for p in peaks_min]
        peak_v = [smoothed[p] for p in peaks_min]
        ax.scatter(peak_t, peak_v, c="blue", s=20, zorder=5, label=f"min ({len(peaks_min)})")
    ax.set_title("Peak Detection")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_extraction{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Plot] Saved: {path}")
