"""
demos/radar_insight_plot.py
用 Insight7 自己画雷达四子图 — 纯 ins + ins.plot
"""

import math
import os, sys
import insight as ins
import numpy as np

PI = math.pi
FS = 10e6
TAU = 50e-6
B = 5e6
N = 1000
T_PRF = 1e-4
N_PULSES = 32
SNR_DB = 10
TARGET_DELAYS = [35e-6, 50e-6]
TARGET_DOPPLERS = [2000.0, -1500.0]

PC_OFFSET = 0
RANGE_PER_BIN = 3e8 / (2 * FS)


def run_frame():
    ins.set_device(ins.CPUPlace())

    t = ins.arange(N, dtype=ins.float64) / FS
    phase = PI * (B / TAU) * (t ** 2)
    s_tx = ins.to_complex(ins.cos(phase), ins.sin(phase))
    tau_arr = ins.full([1], TAU, ins.float64)
    mask = ins.cast(ins.less(t, tau_arr), ins.float64)
    s_tx = s_tx * mask

    sig_power = float(ins.mean(ins.real(s_tx) ** 2 + ins.imag(s_tx) ** 2).numpy().item())
    noise_sigma = math.sqrt(sig_power / 10 ** (SNR_DB / 10) / 2)

    ins.seed(42)
    noise_r = ins.randn([N_PULSES, N], ins.float64) * noise_sigma
    noise_i = ins.randn([N_PULSES, N], ins.float64) * noise_sigma
    s_rx = ins.zeros([N_PULSES, N], ins.complex128)

    for p in range(N_PULSES):
        slow_time = p * T_PRF
        pulse = ins.zeros([N], ins.complex128)
        for delay, doppler in zip(TARGET_DELAYS, TARGET_DOPPLERS):
            ds = int(delay * FS)
            delayed = ins.zeros([N], ins.complex128)
            if ds < N:
                delayed[ds:] = s_tx[0:N - ds]
            fast_phase = 2 * PI * doppler * t
            rotation = ins.to_complex(ins.cos(fast_phase), ins.sin(fast_phase))
            slow_phase = 2 * PI * doppler * slow_time
            slow_rot = ins.to_complex(
                ins.full([1], math.cos(slow_phase), ins.float64),
                ins.full([1], math.sin(slow_phase), ins.float64),
            )
            rotation = rotation * slow_rot
            pulse = pulse + delayed * rotation
        pulse = pulse + ins.to_complex(noise_r[p], noise_i[p])
        s_rx[p] = pulse

    mf = ins.conj(ins.flip(s_tx))
    pc = ins.zeros([N_PULSES, N], ins.complex128)
    for p in range(N_PULSES):
        conv = ins.signal.fftconvolve(s_rx[p], mf, "full")
        start = N - 1
        pc[p] = conv[start:start + N]

    doppler_spec = ins.fft(pc, N_PULSES, 0)
    doppler_shifted = ins.fftshift(doppler_spec, 0)

    energy = ins.real(doppler_shifted) ** 2 + ins.imag(doppler_shifted) ** 2
    energy_db = 10 * ins.log10(energy + 1e-12)

    _, det = ins.signal.ca_cfar(energy, [2, 2], [4, 4], 1e-5)

    det_np = det.numpy().astype(bool)
    idx = det_np.nonzero()
    targets = []
    if len(idx[0]) > 0:
        points = [(int(idx[0][i]), int(idx[1][i])) for i in range(len(idx[0]))]
        visited = [False] * len(points)
        for i in range(len(points)):
            if visited[i]:
                continue
            visited[i] = True
            cluster = [points[i]]
            for j in range(i + 1, len(points)):
                if visited[j]:
                    continue
                d = math.sqrt((points[i][0] - points[j][0]) ** 2 +
                              (points[i][1] - points[j][1]) ** 2)
                if d <= 5.0:
                    visited[j] = True
                    cluster.append(points[j])
            cd = int(round(sum(p[0] for p in cluster) / len(cluster)))
            cr = int(round(sum(p[1] for p in cluster) / len(cluster)))
            targets.append((cd, cr))

    doppler_bins = ins.fftshift(ins.fftfreq(N_PULSES, T_PRF))
    range_bins = (ins.arange(N, dtype=ins.float64) - PC_OFFSET) * RANGE_PER_BIN

    return {
        "pc": pc,
        "energy_db": energy_db,
        "energy": energy,
        "targets": targets,
        "doppler_bins": doppler_bins,
        "range_bins": range_bins,
    }


def save_plots(r, path):
    plt = ins.plot
    pc = r["pc"]
    energy = r["energy"]
    energy_db = r["energy_db"]
    targets = r["targets"]
    db_np = r["doppler_bins"].numpy()
    rb_np = r["range_bins"].numpy()

    pc_abs = ins.abs(pc)
    max_r = int(ins.argmax(ins.max(energy, axis=1)).numpy().item())

    plt.figure()

    # (1) Pulse Compression
    plt.subplot(2, 2, 1)
    plt.image(pc_abs)
    plt.title("Pulse Compression")
    plt.xlabel("Range Gate")
    plt.ylabel("Pulse #")
    plt.colorbar()

    # (2) Range-Doppler Map 
    plt.subplot(2, 2, 2)
    plt.image(energy_db)
    plt.colormap(plt.Colormap.jet)
    plt.title("Range-Doppler Map")
    plt.xlabel("Range [m]")
    plt.ylabel("Doppler [Hz]")
    plt.colorbar()

    # (3) CFAR Detection + Target markers
    plt.subplot(2, 2, 3)
    plt.image(energy_db)
    plt.colormap(plt.Colormap.jet)
    plt.title(f"CFAR ({len(targets)} targets)")
    plt.xlabel("Range [m]")
    plt.ylabel("Doppler [Hz]")
    if targets:
        tx = ins.zeros([len(targets)], ins.float64)
        ty = ins.zeros([len(targets)], ins.float64)
        for i, (d, rr) in enumerate(targets):
            if 0 <= d < len(db_np) and 0 <= rr < len(rb_np):
                tx[i] = rb_np[rr]
                ty[i] = db_np[d]
        plt.scatter(tx, ty)
    plt.colorbar()

    # (4) Doppler spectrum at max range gate
    plt.subplot(2, 2, 4)
    plt.plot(r["doppler_bins"], energy[:, max_r])
    plt.title(f"Doppler Spectrum")
    plt.xlabel("Doppler [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.save(path)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    ins.init()
    out = "/home/aistudio/plum/Insight7/demos/python/output"
    os.makedirs(out, exist_ok=True)

    print("Running Insight7 radar + plotting...")
    result = run_frame()
    print(f"  Targets: {len(result['targets'])}")

    outpath = os.path.join(out, "radar_insight_plot.png")
    save_plots(result, outpath)
    print("Done!")
