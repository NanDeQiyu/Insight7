# demos/radar_task1.jl
# 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Julia 版)

using Insight

println("=" ^ 60)
println("  雷达目标检测与多普勒分析 (Insight7 Julia)")
println("=" ^ 60)

# ========== 参数配置 ==========
fs = 10e6           # 采样率 10 MHz
T = 1e-4            # 脉冲重复周期 100 us
N = 1000            # 单脉冲采样点数
B = 5e6             # 带宽 5 MHz
tau = 50e-6         # 脉宽 50 us
n_pulses = 128
SNR_dB = 10
target_delays = [30e-6, 45e-6]
target_dopplers = [2000.0, -1500.0]

range_res = 3e8 / (2 * B)

println("\n[配置信息]")
println("  采样率: $(fs/1e6) MHz")
println("  信号带宽: $(B/1e6) MHz")
println("  单脉冲采样点数: $N")
println("  脉冲串数量: $n_pulses")
println("  距离分辨率: $(round(range_res, digits=2)) 米")

# ========== 生成 LFM 发射信号 ==========
t = collect(0:N-1) ./ fs
s_tx = Complex{Float64}.(exp.(1im * π * (B/tau) .* t.^2) .* (t .< tau))

sig_power = mean(abs.(s_tx).^2)
noise_power = sig_power / 10^(SNR_dB/10)
noise_sigma = sqrt(noise_power / 2)

# ========== 模拟回波 ==========
println("\n[1/5] 模拟回波信号...")

rng = MersenneTwister(42)
s_rx = zeros(Complex{Float64}, n_pulses, N)

for p in 1:n_pulses
    slow_time = (p - 1) * T
    pulse = zeros(Complex{Float64}, N)

    for (delay, doppler) in zip(target_delays, target_dopplers)
        ds = Int(delay * fs) + 1  # 1-based
        if ds <= N
            pulse[ds:N] .+= s_tx[1:N-ds+1]
        end
    end

    # Fast-time Doppler
    pulse .*= exp.(1im * 2π * target_dopplers[1] .* t)

    # Slow-time Doppler
    pulse .*= exp(1im * 2π * target_dopplers[1] * slow_time)

    # Noise
    noise = noise_sigma .* (randn(rng, N) .+ 1im .* randn(rng, N))
    s_rx[p, :] .= pulse .+ noise
end

println("  回波矩阵维度: $(size(s_rx))")

# ========== 脉冲压缩 ==========
println("\n[2/5] 脉冲压缩...")
t0 = time()

matched_filter = conj(reverse(s_tx))
mf_ins = Insight.from_numpy(collect(matched_filter))

pc = zeros(Complex{Float64}, n_pulses, N)
for p in 1:n_pulses
    pulse_ins = Insight.from_numpy(collect(s_rx[p, :]))
    conv = Insight.signal.fftconvolve(pulse_ins, mf_ins, "full")
    conv_jl = Insight.numpy(conv)
    full_len = length(conv_jl)
    start = div(full_len - N, 2) + 1
    pc[p, :] .= conv_jl[start:start+N-1]
end

println("  耗时: $(round(time() - t0, digits=4)) 秒")

# ========== 多普勒处理 ==========
println("\n[3/5] 多普勒处理...")
t0 = time()

pc_ins = Insight.from_numpy(collect(pc))
doppler_fft = Insight.fft.fft(pc_ins, n_pulses)
doppler_np = Insight.numpy(doppler_fft)
doppler_np = fftshift(doppler_np, 1)

println("  耗时: $(round(time() - t0, digits=4)) 秒")

# ========== CA-CFAR 检测 ==========
println("\n[4/5] CFAR 目标检测...")
t0 = time()

energy = abs.(doppler_np)
energy_ins = Insight.from_numpy(energy)

threshold, detections = Insight.signal.ca_cfar(energy_ins, [2, 2], [4, 4], 1e-5)
det_np = Bool.(Insight.numpy(detections))

println("  耗时: $(round(time() - t0, digits=4)) 秒")

# ========== 目标提取 ==========
println("\n[5/5] 提取目标...")

target_indices = findall(det_np)
raw_count = length(target_indices)

# Simple clustering
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

println("\n[检测结果]")
println("  原始检测点数: $raw_count")
println("  聚类后目标数: $(length(targets))")

doppler_bins = fftshift(fftfreq(n_pulses, 1/T))
range_bins = collect(0:N-1) .* (3e8 / (2 * fs))

if !isempty(targets)
    println("\n  检测到的目标:")
    for (d_idx, r_idx) in targets
        if 1 <= d_idx <= length(doppler_bins) && 1 <= r_idx <= length(range_bins)
            println("    → 距离: $(round(range_bins[r_idx], digits=2)) 米, " *
                    "多普勒: $(round(doppler_bins[d_idx], digits=1)) Hz, " *
                    "强度: $(round(energy[d_idx, r_idx], digits=3))")
        end
    end
end

println("\n完成！")
