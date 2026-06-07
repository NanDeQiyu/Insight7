"""
_plot_radar.py
雷达检测 matplotlib 绘图模块。仅被 radar_detection.py 的 --plot 分支导入。
允许使用 numpy 和 matplotlib。
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_frame(r, output_dir, frame_idx=None):
    """四子图，支持帧序号命名。"""
    pc = r["pc"].numpy()
    energy = r["energy"].numpy()
    energy_db = 20 * np.log10(np.abs(energy) + 1e-8)
    targets = r["targets"]
    db = np.array(r["doppler_bins"])
    rb = np.array(r["range_bins"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 脉冲压缩
    ax = axes[0, 0]
    ax.imshow(np.abs(pc), aspect="auto", cmap="viridis")
    ax.set_title("Pulse Compression")
    ax.set_xlabel("Range Gate")
    ax.set_ylabel("Pulse #")

    # (2) 距离-多普勒谱
    ax = axes[0, 1]
    ax.imshow(
        energy_db,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[rb[0], rb[-1], db[0], db[-1]],
    )
    ax.set_title("Range-Doppler Map")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Doppler [Hz]")

    # (3) CFAR 检测 + 红色十字
    ax = axes[1, 0]
    ax.imshow(
        energy_db,
        aspect="auto",
        cmap="inferno",
        origin="lower",
        extent=[rb[0], rb[-1], db[0], db[-1]],
    )
    if targets:
        rs = [rb[rr] for _, rr in targets]
        ds = [db[d] for d, _ in targets]
        ax.scatter(rs, ds, c="red", s=200, marker="x", linewidths=3, label="Target")
        ax.legend(fontsize=9)
    ax.set_title(f"Frame {frame_idx} — {len(targets)} targets")
    ax.set_xlabel("Range [m]")
    ax.set_ylabel("Doppler [Hz]")

    # (4) 多普勒谱
    ax = axes[1, 1]
    max_r = int(np.argmax(np.max(np.abs(energy), axis=0)))
    ax.plot(db, np.abs(energy[:, max_r]))
    ax.set_title(f"Doppler Spectrum (range {rb[max_r]:.0f}m)")
    ax.set_xlabel("Doppler [Hz]")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if frame_idx is not None:
        name = f"frame_{frame_idx:04d}.png"
    else:
        name = f"radar_{r['device']}.png"
    path = os.path.join(output_dir, name)
    plt.savefig(path, dpi=150)
    plt.close(fig)


def save_ambiguity_plot(ambg, output_dir="."):
    """模糊函数图。"""
    ambg_np = ambg.numpy()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(20 * np.log10(np.abs(ambg_np) + 1e-12), aspect="auto", cmap="jet")
    ax.set_title("Ambiguity Function")
    ax.set_xlabel("Delay")
    ax.set_ylabel("Doppler")
    plt.colorbar(ax.imshow(20 * np.log10(np.abs(ambg_np) + 1e-12), aspect="auto", cmap="jet"))
    plt.tight_layout()
    name = f"radar_{os.path.basename(output_dir)}_ambiguity.png"
    path = os.path.join(output_dir, name)
    plt.savefig(path, dpi=150)
    plt.close(fig)
