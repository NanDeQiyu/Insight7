"""
_plot_pipeline.py
信号分析流水线 matplotlib 绘图模块。
仅被 signal_pipeline.py 的 --plot 分支导入。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plots(r, output_dir="."):
    """六子图可视化。"""
    wavs = r["wavs"]
    features = r["features"]
    t = wavs["t"].numpy()
    fs = r["fs"]

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
    ax.plot(t, r["signal_filtered"].numpy(), alpha=0.6, label="filtered")
    ax.plot(t, r["signal_smooth"].numpy(), linewidth=2, label="smoothed")
    ax.set_title("Composite → Filtered → Smoothed")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (3) 频谱图
    ax = axes[1, 0]
    Sxx = features["spec_Sxx"].numpy()
    f_spec = features["spec_f"].numpy()
    t_spec = features["spec_t"].numpy()
    # Sxx shape is (n_freq, n_time), pcolormesh expects C[Y, X]
    ax.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx.T + 1e-12), shading="gouraud", cmap="inferno")
    ax.set_title("Spectrogram (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")

    # (4) CWT 尺度图
    ax = axes[1, 1]
    cwt = np.abs(features["cwt_matrix"].numpy())
    ax.imshow(
        cwt,
        aspect="auto",
        cmap="viridis",
        extent=[t[0], t[-1], features["cwt_widths"][-1], features["cwt_widths"][0]],
    )
    ax.set_title("CWT Scalogram (morlet2)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Scale")

    # (5) FM 解调 + Kalman
    ax = axes[2, 0]
    fm = features["fm_demod"].numpy()
    ax.plot(t[: len(fm)], fm, alpha=0.4, label="FM demod")
    smoothed = r["smoothed_freq"]
    if smoothed:
        ax.plot(t[: len(smoothed)], smoothed, linewidth=2, label="Kalman")
    params = r["params"]
    if params:
        pt = [p["time"] for p in params]
        pf = [p["inst_freq"] for p in params]
        ax.scatter(pt, pf, c="red", s=30, zorder=5, label="peaks")
    ax.set_title("FM Demod + Kalman Estimation")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous Freq (Hz)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (6) 自相关
    ax = axes[2, 1]
    ac = features["autocorr"].numpy()
    n_ac = min(500, len(ac))
    lags = np.arange(n_ac) / fs
    ax.plot(lags, ac[:n_ac])
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lag (s)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"signal_pipeline_{r['device']}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已保存: {path}")
