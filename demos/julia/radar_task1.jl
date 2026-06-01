# demos/radar_task1.jl
# 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Julia 完整版)
# 与 C++ / Python / Lua 版本功能完全对齐

using Insight

# ========== 参数 ==========
const FS = 10e6
const T_PRF = 1e-4
const N = 1000
const B = 5e6
const TAU = 50e-6
const N_PULSES = 128
const SNR_DB = 10
const TARGET_DELAYS = [30e-6, 45e-6]
const TARGET_DOPPLERS = [2000.0, -1500.0]
const RANGE_RES = 3e8 / (2 * B)
const RANGE_PER_BIN = 3e8 / (2 * FS)
const PC_OFFSET = div(N - 1, 2)

println("=" ^ 60)
println("  雷达目标检测与多普勒分析 (Insight7 Julia)")
println("=" ^ 60)

println("\n[配置信息]")
println("  采样率: $(FS/1e6) MHz, 带宽: $(B/1e6) MHz")
println("  单脉冲采样点数: $N, 脉冲串数量: $N_PULSES")
println("  距离分辨率: $(round(RANGE_RES, digits=2)) 米, 距离门: $(round(RANGE_PER_BIN, digits=2)) 米")

# ========== 1. LFM 发射信号 ==========
println("\n[1/6] 生成 LFM 信号...")
t = collect(0:N-1) ./ FS
s_tx = Complex{Float64}.(exp.(1im * π * (B/TAU) .* t.^2) .* (t .< TAU))
sig_power = mean(abs.(s_tx).^2)
noise_sigma = sqrt(sig_power / 10^(SNR_DB/10) / 2)

# ========== 2. 模拟回波 ==========
println("[2/6] 模拟回波信号...")
rng = MersenneTwister(42)
s_rx = zeros(Complex{Float64}, N_PULSES, N)

for p in 1:N_PULSES
    slow_time = (p - 1) * T_PRF
    pulse = zeros(Complex{Float64}, N)

    for (delay, doppler) in zip(TARGET_DELAYS, TARGET_DOPPLERS)
        tgt = zeros(Complex{Float64}, N)
        ds = Int(delay * FS) + 1
        if ds <= N
            tgt[ds:N] .= s_tx[1:N-ds+1]
        end
        tgt .*= exp.(1im * 2π * doppler .* t)
        tgt .*= exp(1im * 2π * doppler * slow_time)
        pulse .+= tgt
    end

    noise = noise_sigma .* (randn(rng, N) .+ 1im .* randn(rng, N))
    s_rx[p, :] .= pulse .+ noise
end

# ========== 3. 脉冲压缩 ==========
println("[3/6] 脉冲压缩...")
t0 = time()

mf = conj(reverse(s_tx))
mf_ins = Insight.from_numpy(collect(mf))

pc = zeros(Complex{Float64}, N_PULSES, N)
for p in 1:N_PULSES
    pulse_ins = Insight.from_numpy(collect(s_rx[p, :]))
    conv = Insight.signal.fftconvolve(pulse_ins, mf_ins, "full")
    conv_jl = Insight.numpy(conv)
    full_len = length(conv_jl)
    start = div(full_len - N, 2) + 1
    pc[p, :] .= conv_jl[start:start+N-1]
end

println("  耗时: $(round(time() - t0, digits=2)) 秒")

# ========== 4. 多普勒 FFT ==========
println("[4/6] 多普勒处理...")
t0 = time()

pc_ins = Insight.from_numpy(collect(pc))
doppler_fft = Insight.fft(pc_ins, N_PULSES, 0)
doppler_np = Insight.numpy(doppler_fft)
doppler_np = fftshift(doppler_np, 1)

println("  耗时: $(round(time() - t0, digits=2)) 秒")

# ========== 5. CA-CFAR 检测 ==========
println("[5/6] CFAR 目标检测...")
t0 = time()

energy = abs.(doppler_np)
energy_ins = Insight.from_numpy(energy)

threshold, detections = Insight.signal.ca_cfar(energy_ins, [2, 2], [4, 4], 1e-5)
det_np = Bool.(Insight.numpy(detections))

println("  耗时: $(round(time() - t0, digits=2)) 秒")

# ========== 6. 目标聚类 ==========
println("[6/6] 聚类目标...")

target_indices = findall(det_np)
raw_count = length(target_indices)

visited = falses(raw_count)
targets = Tuple{Int,Int}[]

for i in 1:raw_count
    if visited[i]
        continue
    end
    visited[i] = true
    cluster = [target_indices[i]]
    for j in i+1:raw_count
        if visited[j]
            continue
        end
        d = sqrt((target_indices[i][1] - target_indices[j][1])^2 +
                 (target_indices[i][2] - target_indices[j][2])^2)
        if d <= 5.0
            visited[j] = true
            push!(cluster, target_indices[j])
        end
    end
    center_d = round(Int, mean(c[1] for c in cluster))
    center_r = round(Int, mean(c[2] for c in cluster))
    push!(targets, (center_d, center_r))
end

# ========== 输出 ==========
doppler_bins = fftshift(fftfreq(N_PULSES, 1/T_PRF))

println("\n[检测结果]")
println("  原始检测点数: $raw_count, 聚类后: $(length(targets))")

for (d_idx, r_idx) in targets
    range_m = (r_idx - PC_OFFSET) * RANGE_PER_BIN
    doppler_hz = doppler_bins[d_idx]
    println("    → 距离: $(lpad(round(range_m, digits=2), 8)) 米, " *
            "多普勒: $(lpad(round(doppler_hz, digits=1), 8)) Hz")
end

println("\n完成！")
