"""
demos/radar_task1.py
雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Python 版)
与 C++ / Lua / Julia 版本功能完全对齐
"""

import time
import numpy as np
import insight as ins


# ========== 参数 ==========
FS = 10e6
T_PRF = 1e-4
N = 1000
B = 5e6
TAU = 50e-6
N_PULSES = 128
SNR_DB = 10
TARGET_DELAYS = [30e-6, 45e-6]
TARGET_DOPPLERS = [2000.0, -1500.0]
RANGE_RES = 3e8 / (2 * B)


def run_task1(device="cpu"):
    if device == "gpu":
        ins.load_backend("cuda")
        place = ins.GPUPlace(0)
    else:
        place = ins.CPUPlace()

    t = np.arange(N) / FS

    # 1. LFM 发射信号
    s_tx = np.exp(1j * np.pi * (B / TAU) * t**2).astype(np.complex128)
    s_tx[t >= TAU] = 0.0
    sig_power = np.mean(np.abs(s_tx) ** 2)
    noise_sigma = np.sqrt(sig_power / 10 ** (SNR_DB / 10) / 2)

    # 2. 模拟回波 (逐目标施加 Doppler，与 C++ 对齐)
    rng = np.random.RandomState(42)
    s_rx = np.zeros((N_PULSES, N), dtype=np.complex128)

    for p in range(N_PULSES):
        slow_time = p * T_PRF
        pulse = np.zeros(N, dtype=np.complex128)
        for delay, doppler in zip(TARGET_DELAYS, TARGET_DOPPLERS):
            ds = int(delay * FS)
            tgt = np.zeros(N, dtype=np.complex128)
            if ds < N:
                tgt[ds:] = s_tx[: N - ds]
            tgt *= np.exp(1j * 2 * np.pi * doppler * t)
            tgt *= np.exp(1j * 2 * np.pi * doppler * slow_time)
            pulse += tgt
        pulse += noise_sigma * (rng.randn(N) + 1j * rng.randn(N))
        s_rx[p, :] = pulse

    # 3. 脉冲压缩
    mf = np.conj(s_tx[::-1])
    mf_ins = ins.from_numpy(mf)
    t0 = time.time()
    pc = np.zeros((N_PULSES, N), dtype=np.complex128)
    for p in range(N_PULSES):
        conv = ins.signal.fftconvolve(ins.from_numpy(s_rx[p, :]), mf_ins, "full")
        c = conv.numpy()
        start = (len(c) - N) // 2
        pc[p, :] = c[start : start + N]
    t_pc = time.time() - t0

    # 4. 多普勒 FFT
    t0 = time.time()
    doppler_fft = ins.fft(ins.from_numpy(pc), N_PULSES, 0)
    doppler_np = np.fft.fftshift(doppler_fft.numpy(), axes=0)
    t_doppler = time.time() - t0

    # 5. CA-CFAR
    energy = np.abs(doppler_np)
    t0 = time.time()
    _, det = ins.signal.ca_cfar(ins.from_numpy(energy), [2, 2], [4, 4], 1e-5)
    det_np = det.numpy().astype(bool)
    t_cfar = time.time() - t0

    # 6. 聚类
    target_indices = np.argwhere(det_np)
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
            if np.sqrt(np.sum((target_indices[i] - target_indices[j]) ** 2)) <= 5:
                visited[j] = True
                cluster.append(target_indices[j])
        center = np.mean(cluster, axis=0)
        targets.append((int(round(center[0])), int(round(center[1]))))

    total_ms = (t_pc + t_doppler + t_cfar) * 1000
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(N_PULSES, T_PRF))
    range_bins = np.arange(N) * (3e8 / (2 * FS))

    return {
        "targets": targets,
        "raw_count": len(target_indices),
        "total_ms": total_ms,
        "pc_ms": t_pc * 1000,
        "doppler_ms": t_doppler * 1000,
        "cfar_ms": t_cfar * 1000,
        "doppler_bins": doppler_bins,
        "range_bins": range_bins,
        "energy": energy,
        "device": device,
    }


def print_result(r):
    dev = r["device"].upper()
    print(
        f"\n  {dev} 耗时: {r['total_ms']:.2f} ms "
        f"(PC: {r['pc_ms']:.1f}, Doppler: {r['doppler_ms']:.1f}, CFAR: {r['cfar_ms']:.1f})"
    )
    print(f"  原始检测点数: {r['raw_count']}, 聚类后: {len(r['targets'])}")
    for d, rr in r["targets"]:
        if 0 <= d < len(r["doppler_bins"]) and 0 <= rr < len(r["range_bins"]):
            print(
                f"    → 距离: {r['range_bins'][rr]:7.2f} 米, "
                f"多普勒: {r['doppler_bins'][d]:8.1f} Hz, "
                f"强度: {r['energy'][d, rr]:.3f}"
            )


def save_plots(r, prefix):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 距离-多普勒谱
        ax = axes[0, 0]
        db = 20 * np.log10(r["energy"] + 1e-8)
        ax.imshow(
            db,
            aspect="auto",
            cmap="inferno",
            extent=[
                r["range_bins"][0],
                r["range_bins"][-1],
                r["doppler_bins"][-1],
                r["doppler_bins"][0],
            ],
        )
        ax.set_title("Range-Doppler Map")
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Doppler [Hz]")
        plt.colorbar(ax.images[0], ax=ax, label="dB")

        # CFAR 检测叠加
        ax = axes[0, 1]
        ax.imshow(
            db,
            aspect="auto",
            cmap="inferno",
            extent=[
                r["range_bins"][0],
                r["range_bins"][-1],
                r["doppler_bins"][-1],
                r["doppler_bins"][0],
            ],
        )
        for d, rr in r["targets"]:
            if 0 <= d < len(r["doppler_bins"]) and 0 <= rr < len(r["range_bins"]):
                ax.scatter(
                    r["range_bins"][rr],
                    r["doppler_bins"][d],
                    c="red",
                    s=100,
                    marker="x",
                    linewidths=3,
                )
        ax.set_title("CFAR Detection")
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Doppler [Hz]")

        # 多普勒切面
        max_r = np.argmax(np.max(r["energy"], axis=0))
        ax = axes[1, 0]
        ax.plot(r["doppler_bins"], r["energy"][:, max_r])
        ax.set_title(f"Doppler Spectrum (range bin {max_r})")
        ax.set_xlabel("Doppler [Hz]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        # 距离切面
        max_d = np.argmax(np.max(r["energy"], axis=1))
        ax = axes[1, 1]
        ax.plot(r["range_bins"], r["energy"][max_d, :])
        ax.set_title(f"Range Profile (doppler bin {max_d})")
        ax.set_xlabel("Range [m]")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{prefix}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  已保存: {path}")
    except Exception as e:
        print(f"  绘图失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("  雷达目标检测与多普勒分析 (Insight7 Python)")
    print("=" * 60)

    print("\n[配置信息]")
    print(f"  采样率: {FS/1e6:.1f} MHz, 带宽: {B/1e6:.1f} MHz")
    print(f"  单脉冲采样点数: {N}, 脉冲串数量: {N_PULSES}")
    print(f"  距离分辨率: {RANGE_RES:.2f} 米")

    ins.init(["cpu"])
    print("\n" + "=" * 60 + "\n  CPU 运行\n" + "=" * 60)
    cpu = run_task1("cpu")
    print_result(cpu)
    save_plots(cpu, "task1_cpu")

    try:
        ins.load_backend("cuda")
        print("\n" + "=" * 60 + "\n  GPU 运行\n" + "=" * 60)
        gpu = run_task1("gpu")
        print_result(gpu)
        save_plots(gpu, "task1_gpu")

        print("\n" + "=" * 60 + "\n  性能对比\n" + "=" * 60)
        speedup = cpu["total_ms"] / gpu["total_ms"]
        print(f"  CPU: {cpu['total_ms']:.2f} ms, GPU: {gpu['total_ms']:.2f} ms")
        print(f"  加速比: {speedup:.2f}x {'✅ GPU 更快' if speedup > 1 else '⚠️ 需优化'}")
    except Exception as e:
        print(f"\n  GPU 不可用: {e}")

    print("\n完成！")
