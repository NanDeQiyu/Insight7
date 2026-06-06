"""
demos/radar_detection.py
雷达目标检测与多普勒分析 (Insight7)
脉冲压缩 · Doppler FFT · CA-CFAR · 模糊函数分析
"""

import argparse
import os
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
RANGE_PER_BIN = 3e8 / (2 * FS)
PC_OFFSET = (N - 1) // 2


def run_detection(device="cpu"):
    if device == "gpu" and ins.has_device("gpu"):
        ins.load_backend("cuda")
        ins.set_device(ins.GPUPlace(0))
    else:
        if device == "gpu":
            print("  [警告] GPU 不可用，回退到 CPU")
        ins.set_device(ins.CPUPlace())
        device = "cpu"

    t = np.arange(N) / FS

    # 1. LFM 发射信号
    s_tx = np.exp(1j * np.pi * (B / TAU) * t**2).astype(np.complex128)
    s_tx[t >= TAU] = 0.0
    sig_power = np.mean(np.abs(s_tx) ** 2)
    noise_sigma = np.sqrt(sig_power / 10 ** (SNR_DB / 10) / 2)

    # 2. 模拟回波
    ins.seed(42)
    noise_r = (ins.randn([N_PULSES, N], ins.float64) * noise_sigma).numpy()
    noise_i = (ins.randn([N_PULSES, N], ins.float64) * noise_sigma).numpy()
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
        pulse += noise_r[p, :] + 1j * noise_i[p, :]
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
    doppler_shifted = ins.fftshift(doppler_fft, 0)
    t_doppler = time.time() - t0

    # 5. CA-CFAR
    energy_arr = ins.abs(doppler_shifted)
    t0 = time.time()
    _, det = ins.signal.ca_cfar(energy_arr, [2, 2], [4, 4], 1e-5)
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
    range_bins = (np.arange(N) - PC_OFFSET) * RANGE_PER_BIN

    return {
        "targets": targets,
        "raw_count": len(target_indices),
        "total_ms": total_ms,
        "pc_ms": t_pc * 1000,
        "doppler_ms": t_doppler * 1000,
        "cfar_ms": t_cfar * 1000,
        "doppler_bins": doppler_bins,
        "range_bins": range_bins,
        "energy": energy_arr,
        "s_tx": s_tx,
        "device": device,
    }


def run_ambiguity_analysis(s_tx):
    """Compute and return ambiguity function using Insight7."""
    s_tx_ins = ins.from_numpy(s_tx.astype(np.complex128))
    t0 = time.time()
    ambg = ins.signal.ambgfun(s_tx_ins, FS, 1.0 / T_PRF, cut="2d")
    t_ambg = time.time() - t0
    return ambg, t_ambg


def print_result(r):
    dev = r["device"].upper()
    print(
        f"\n  {dev} 耗时: {r['total_ms']:.2f} ms "
        f"(PC: {r['pc_ms']:.1f}, Doppler: {r['doppler_ms']:.1f}, CFAR: {r['cfar_ms']:.1f})"
    )
    print(f"  原始检测点数: {r['raw_count']}, 聚类后: {len(r['targets'])}")
    energy_np = r["energy"].numpy()
    if energy_np.dtype.kind == "c":
        energy_np = np.abs(energy_np)
    for d, rr in r["targets"]:
        if 0 <= d < len(r["doppler_bins"]) and 0 <= rr < len(r["range_bins"]):
            print(
                f"    → 距离: {r['range_bins'][rr]:7.2f} 米, "
                f"多普勒: {r['doppler_bins'][d]:8.1f} Hz"
            )


def save_plots(r, prefix, output_dir):
    """Save radar analysis plots."""
    try:
        plt = ins.plot
        energy_np = r["energy"].numpy()
        if energy_np.dtype.kind == "c":
            energy_np = np.abs(energy_np)
        db = 20 * np.log10(energy_np + 1e-8)

        plt.figure()
        rr, dd = np.meshgrid(r["range_bins"], r["doppler_bins"])
        plt.contour(
            ins.from_numpy(rr.astype(np.float64)),
            ins.from_numpy(dd.astype(np.float64)),
            ins.from_numpy(db.astype(np.float64)),
        )
        plt.title("Range-Doppler Map")
        plt.xlabel("Range [m]")
        plt.ylabel("Doppler [Hz]")
        path = os.path.join(output_dir, f"{prefix}_range_doppler.png")
        plt.save(path)

        max_r = int(np.argmax(np.max(energy_np, axis=0)))
        doppler_arr = ins.from_numpy(r["doppler_bins"].astype(np.float64))
        ds_arr = ins.from_numpy(energy_np[:, max_r].astype(np.float64))
        plt.figure()
        plt.plot(doppler_arr, ds_arr)
        plt.title(f"Doppler Spectrum (range bin {max_r})")
        plt.xlabel("Doppler [Hz]")
        plt.ylabel("Amplitude")
        plt.grid(True)
        path = os.path.join(output_dir, f"{prefix}_doppler_slice.png")
        plt.save(path)

        print(f"  已保存: {prefix}_*.png")
    except Exception as e:
        print(f"  绘图失败: {e}")


def save_ambiguity_plot(ambg, prefix, output_dir):
    """Save ambiguity function plot."""
    try:
        plt = ins.plot
        ambg_np = ambg.numpy()
        if ambg_np.dtype.kind == "c":
            ambg_np = np.abs(ambg_np)
        ambg_db = 20 * np.log10(ambg_np + 1e-12)

        plt.figure()
        plt.imshow(ins.from_numpy(ambg_db.astype(np.float64)))
        plt.title("Ambiguity Function (2D)")
        plt.xlabel("Delay")
        plt.ylabel("Doppler")
        plt.colorbar()
        path = os.path.join(output_dir, f"{prefix}_ambiguity.png")
        plt.save(path)
        print(f"  已保存: {prefix}_ambiguity.png")
    except Exception as e:
        print(f"  模糊函数绘图失败: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="雷达目标检测与多普勒分析 (Insight7)")
    parser.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu", help="运行设备 (默认: cpu)"
    )
    parser.add_argument("--images", type=int, default=0, help="生成 N 张图并报告 FPS (0=不生成)")
    parser.add_argument("--benchmark", action="store_true", help="纯计时模式 (不画图)")
    parser.add_argument("--output", type=str, default=".", help="图片输出目录 (默认: 当前目录)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  雷达目标检测与多普勒分析 (Insight7)")
    print("=" * 60)

    print("\n[配置信息]")
    print(f"  采样率: {FS/1e6:.1f} MHz, 带宽: {B/1e6:.1f} MHz")
    print(f"  单脉冲采样点数: {N}, 脉冲串数量: {N_PULSES}")
    print(f"  距离分辨率: {RANGE_RES:.2f} 米")
    print(f"  设备: {args.device}")

    ins.init()

    os.makedirs(args.output, exist_ok=True)

    # --- 核心检测 ---
    print("\n" + "=" * 60)
    print(f"  {args.device.upper()} 运行")
    print("=" * 60)

    result = run_detection(args.device)
    print_result(result)

    if not args.benchmark:
        save_plots(result, f"radar_{args.device}", args.output)

    # --- 模糊函数分析 ---
    print("\n[模糊函数分析]")
    ambg, t_ambg = run_ambiguity_analysis(result["s_tx"])
    print(f"  耗时: {t_ambg:.4f} 秒")

    if not args.benchmark:
        save_ambiguity_plot(ambg, f"radar_{args.device}", args.output)

    # --- FPS 测试 ---
    if args.images > 0:
        print(f"\n[FPS 测试] 生成 {args.images} 张图...")
        times = []
        for i in range(args.images):
            t0 = time.time()
            r = run_detection(args.device)
            if not args.benchmark:
                save_plots(r, f"radar_{args.device}_frame{i}", args.output)
            times.append(time.time() - t0)
        avg = np.mean(times)
        print(f"  平均: {avg*1000:.2f} ms/帧, FPS: {1/avg:.1f}")

    if ins.has_device("gpu") and args.device == "cpu":
        print("\n[提示] GPU 可用，使用 --device gpu 运行 GPU 版本")

    print("\n完成！")
