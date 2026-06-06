"""
demos/radar_animation.py
雷达目标检测动画生成 (scipy/numpy)
两个移动目标 × 200 帧 → 四子图 PNG 序列 → PPT 连续播放

基于 mission_1.py 算法，优化脉冲数，跳过低效步骤。
目标每帧移动，生成连续帧可组装为"实时 AI 雷达"演示。

用法:
    python radar_animation.py                     # 生成 200 帧
    python radar_animation.py --frames 50         # 只生成 50 帧
    python radar_animation.py --frames 10 --dpi 72  # 快速测试 + 低分辨率
"""

import argparse
import os
import sys
import time
import numpy as np
from scipy import signal as sig
from scipy.ndimage import maximum_filter
from scipy.cluster.hierarchy import fclusterdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================
# 参数
# ============================================================
FS = 10e6                     # 采样率 10 MHz
PRI = 100e-6                  # 脉冲重复间隔
TAU = 50e-6                   # 脉冲宽度
B = 5e6                       # 带宽 5 MHz
SNR_DB = 10                   # 信噪比
N_PULSES = 32                 # 每帧脉冲数 (加速至 1-2s/帧)

# 目标初始参数
T1_INIT_DELAY = 35e-6         # 目标1 初始延迟 35us (~5250m 距离)
T2_INIT_DELAY = 50e-6         # 目标2 初始延迟 50us (~7500m 距离)
T1_DOPPLER = 2000             # 目标1 多普勒 Hz
T2_DOPPLER = -1500            # 目标2 多普勒 Hz

# 目标速度 (每帧延迟减少量, us/帧)
T1_SPEED = 0.12e-6            # 35→11us 跨 200 帧
T2_SPEED = 0.15e-6            # 50→20us 跨 200 帧

FRAMES_DIR = "radar_frames"    # 输出目录


def make_t(N):
    """生成时间基。"""
    return np.arange(N) / FS


def build_tx(N, t):
    """生成 LFM 发射信号。"""
    s_tx = np.exp(1j * np.pi * (B / TAU) * t ** 2) * (t < TAU)
    return s_tx


def simulate_echo(s_tx, N, delays, dopplers, slow_time, sigma):
    """模拟单脉冲回波。"""
    pulse = np.zeros(N, dtype=complex)
    for delay, doppler in zip(delays, dopplers):
        ds = int(delay * FS)
        s_target = np.zeros(N, dtype=complex)
        if ds < N:
            s_target[ds:] = s_tx[:N - ds]
        rotation = np.exp(1j * 2 * np.pi * doppler * slow_time)
        pulse += s_target * rotation
    noise = sigma * (np.random.randn(N) + 1j * np.random.randn(N))
    return pulse + noise


def pulse_compress(s_rx, matched_filter):
    """单脉冲脉冲压缩。"""
    return sig.fftconvolve(s_rx, matched_filter, mode='same')


def doppler_fft(pc_all, n_pulses, window):
    """对脉冲串做 Doppler FFT。"""
    spec = np.fft.fft(pc_all * window[:, None], axis=0) / n_pulses
    return np.fft.fftshift(spec, axes=0)


def ca_cfar_2d(energy, guard=(4, 4), ref=(12, 12), pfa=1e-6):
    """2D CA-CFAR 检测。"""
    from scipy.ndimage import uniform_filter
    g_d, g_r = guard
    total_d = g_d + ref[0]
    total_r = g_r + ref[1]
    big = uniform_filter(energy.astype(np.float64),
                         size=(2 * total_d + 1, 2 * total_r + 1),
                         mode='nearest')
    small = uniform_filter(energy.astype(np.float64),
                           size=(2 * g_d + 1, 2 * g_r + 1),
                           mode='nearest')
    n_big = (2 * total_d + 1) * (2 * total_r + 1)
    n_small = (2 * g_d + 1) * (2 * g_r + 1)
    n_ref = n_big - n_small
    if n_ref <= 0:
        return np.zeros_like(energy, dtype=bool)
    noise_est = (big * n_big - small * n_small) / n_ref
    alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1)
    return energy > alpha * noise_est


def extract_targets(detected, energy, range_bins, doppler_bins, top_n=2):
    """从 CFAR 检测结果中提取目标。"""
    local_max = energy == maximum_filter(energy, size=(11, 11))
    thr = np.median(energy) * 100
    valid = detected & local_max & (energy > thr)
    indices = np.argwhere(valid)

    if len(indices) == 0:
        return []

    if len(indices) > 1:
        normalized = indices.astype(float)
        clusters = fclusterdata(normalized, 3.0, criterion='distance')
        uniq = []
        for cid in np.unique(clusters):
            pts = indices[clusters == cid]
            best = max(pts, key=lambda x: energy[x[0], x[1]])
            uniq.append(best)
    else:
        uniq = [tuple(indices[0])]

    # 按能量排序取 top N
    uniq.sort(key=lambda x: energy[x[0], x[1]], reverse=True)
    uniq = uniq[:top_n]

    targets = []
    for d_idx, r_idx in uniq:
        targets.append({
            'range': range_bins[r_idx],
            'doppler': doppler_bins[d_idx],
            'energy': energy[d_idx, r_idx],
            'delay_idx': r_idx,
            'doppler_idx': d_idx,
        })
    return targets


def save_frame(frame_idx, pc_all, energy, range_bins, doppler_bins,
               targets, delays, output_dir, dpi=100):
    """生成四子图并保存为 PNG。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) Pulse compression result
    ax = axes[0, 0]
    ax.imshow(np.abs(pc_all), aspect='auto', cmap='viridis')
    ax.set_title(f'Frame {frame_idx:03d}: Pulse Compression', fontsize=11)
    ax.set_xlabel('Range Gate')
    ax.set_ylabel('Pulse #')

    # (2) Range-Doppler map (inferno = yellow/red)
    ax = axes[0, 1]
    ax.imshow(20 * np.log10(energy + 1e-8), aspect='auto', cmap='inferno',
              origin='lower',
              extent=[range_bins[0], range_bins[-1],
                      doppler_bins[0], doppler_bins[-1]])
    ax.set_title('Range-Doppler Map', fontsize=11)
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Doppler [Hz]')

    # (3) CFAR detection + red cross stars
    ax = axes[1, 0]
    ax.imshow(20 * np.log10(energy + 1e-8), aspect='auto', cmap='inferno',
              origin='lower',
              extent=[range_bins[0], range_bins[-1],
                      doppler_bins[0], doppler_bins[-1]])
    if targets:
        rs = [t['range'] for t in targets]
        ds = [t['doppler'] for t in targets]
        ax.scatter(rs, ds, c='red', s=200, marker='x',
                   linewidths=3, label='Targets')
        # Ground truth (cyan dashed lines)
        for d in delays:
            r_true = (d - (int(TAU * FS) // 2) / FS) * 3e8 / 2
            ax.axvline(r_true, color='cyan', ls='--', alpha=0.3)
    ax.set_title(f'CFAR Detection ({len(targets)} targets)', fontsize=11)
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Doppler [Hz]')
    ax.legend(fontsize=9)

    # (4) Doppler spectrum at max range gate
    ax = axes[1, 1]
    max_r = np.argmax(np.max(energy, axis=0))
    ax.plot(doppler_bins, energy[:, max_r])
    ax.set_title(f'Doppler Spectrum (range {range_bins[max_r]:.0f}m)', fontsize=11)
    ax.set_xlabel('Doppler [Hz]')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='雷达目标检测动画生成')
    parser.add_argument('--frames', type=int, default=200,
                        help='帧数 (默认: 200)')
    parser.add_argument('--output', type=str, default=FRAMES_DIR,
                        help='输出目录')
    parser.add_argument('--dpi', type=int, default=100,
                        help='图片 DPI (默认: 100, 降低可加速)')
    parser.add_argument('--pulses', type=int, default=N_PULSES,
                        help='每帧脉冲数 (默认: 32, 降低可加速)')
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)
    n_pulses = args.pulses
    n_frames = args.frames

    print("=" * 60)
    print("  雷达目标检测动画生成")
    print("=" * 60)
    print(f"\n[配置]")
    print(f"  帧数: {n_frames}")
    print(f"  每帧脉冲: {n_pulses}")
    print(f"  输出目录: {out_dir}")
    print(f"  DPI: {args.dpi}")

    # ---- 准备 ----
    N = int(PRI * FS)       # 1000 采样点
    t = make_t(N)
    pulse_len = int(TAU * FS)
    s_tx = build_tx(N, t)
    matched_filter = np.conj(s_tx[:pulse_len][::-1])

    # 信噪比
    sig_power = np.mean(np.abs(s_tx[:pulse_len]) ** 2)
    noise_sigma = np.sqrt(sig_power / (10 ** (SNR_DB / 10)) / 2)

    # Doppler 参数
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(n_pulses, PRI))
    doppler_window = np.hamming(n_pulses)

    # 距离轴 (米)
    range_bins = (np.arange(N) - pulse_len // 2) * 3e8 / (2 * FS)

    np.random.seed(42)

    total_start = time.time()
    frame_times = []

    print(f"\n[生成 {n_frames} 帧]")

    for frame_idx in range(n_frames):
        t0 = time.time()

        # ----- 目标位置 (逐帧移动) -----
        delay1 = T1_INIT_DELAY - frame_idx * T1_SPEED
        delay2 = T2_INIT_DELAY - frame_idx * T2_SPEED
        delays = [max(delay1, 2e-6), max(delay2, 2e-6)]
        dopplers = [T1_DOPPLER, T2_DOPPLER]

        # ----- 多脉冲回波 + 脉冲压缩 -----
        pc_all = np.zeros((n_pulses, N), dtype=complex)
        for p in range(n_pulses):
            slow_time = p * PRI
            s_rx = simulate_echo(s_tx, N, delays, dopplers, slow_time, noise_sigma)
            pc_all[p] = pulse_compress(s_rx, matched_filter)

        # ----- Doppler FFT -----
        spec = doppler_fft(pc_all, n_pulses, doppler_window)
        energy = np.abs(spec) ** 2

        # ----- CFAR 检测 -----
        detected = ca_cfar_2d(energy)
        targets = extract_targets(detected, energy, range_bins, doppler_bins)

        # ----- 保存帧 -----
        path = save_frame(frame_idx, pc_all, energy, range_bins, doppler_bins,
                          targets, delays, out_dir, dpi=args.dpi)

        elapsed = time.time() - t0
        frame_times.append(elapsed)

        avg = np.mean(frame_times)
        remaining = avg * (n_frames - frame_idx - 1)

        delay_us = [f'{d*1e6:.1f}' for d in delays]
        print(f"  [{frame_idx+1:3d}/{n_frames}] {elapsed:.2f}s "
              f"延迟: {','.join(delay_us)}us "
              f"目标: {len(targets)}  "
              f"剩余: {remaining:.0f}s", flush=True)

    total = time.time() - total_start
    avg_ms = np.mean(frame_times) * 1000
    fps = 1.0 / np.mean(frame_times)

    print(f"\n{'=' * 60}")
    print(f"  生成完成!")
    print(f"  总耗时: {total:.1f}s")
    print(f"  平均: {avg_ms:.0f} ms/帧, FPS: {fps:.2f}")
    print(f"  输出: {out_dir}/")
    print(f"  提示: 在 PPT 中顺序插入 frame_0000.png ~ frame_{n_frames-1:04d}.png")
    print(f"        设置每页自动切换 0.1s 即可看到雷达动画效果")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
