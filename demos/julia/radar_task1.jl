# demos/radar_task1.jl
# 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Julia 完整版)
# 与 C++ / Python / Lua 版本功能完全对齐

push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "bindings", "julia"))
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "build", "bindings", "julia"))
using Insight

# Insight exports shadow some Base functions; use Base.xxx where needed

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
const PC_OFFSET = Base.div(N - 1, 2)

println("=" ^ 60)
println("  雷达目标检测与多普勒分析 (Insight7 Julia)")
println("=" ^ 60)

println("\n[配置信息]")
println("  采样率: $(FS/1e6) MHz, 带宽: $(B/1e6) MHz")
println("  单脉冲采样点数: $N, 脉冲串数量: $N_PULSES")
println("  距离分辨率: $(Base.round(RANGE_RES, digits=2)) 米, 距离门: $(Base.round(RANGE_PER_BIN, digits=2)) 米")

# ========== 1. LFM 发射信号 ==========
println("\n[1/6] 生成 LFM 信号...")
t = collect(0:N-1) ./ FS
s_tx = Complex{Float64}.(exp.(1im * π * (B/TAU) .* t.^2) .* (t .< TAU))
sig_power = sum(Base.abs.(s_tx).^2) / N
noise_sigma = sqrt(sig_power / 10^(SNR_DB/10) / 2)

# ========== 2. 模拟回波 ==========
println("[2/6] 模拟回波信号...")
# 使用 Insight 原生 RNG 确保跨语言一致
Insight.seed(42)
# 生成 1D 噪声数组避免列优先布局问题
noise_r_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * noise_sigma)
noise_i_flat = Insight.to_data(Insight.randn(Int64[N_PULSES * N], Insight.float64) * noise_sigma)
s_rx = Base.zeros(Complex{Float64}, N_PULSES, N)

for p in 1:N_PULSES
    slow_time = (p - 1) * T_PRF
    pulse = Base.zeros(Complex{Float64}, N)

    for (delay, doppler) in zip(TARGET_DELAYS, TARGET_DOPPLERS)
        tgt = Base.zeros(Complex{Float64}, N)
        ds = Int(delay * FS) + 1
        if ds <= N
            tgt[ds:N] .= s_tx[1:N-ds+1]
        end
        tgt .*= exp.(1im * 2π * doppler .* t)
        tgt .*= exp(1im * 2π * doppler * slow_time)
        pulse .+= tgt
    end

    noise = noise_r_flat[(p-1)*N+1 : p*N] .+ 1im .* noise_i_flat[(p-1)*N+1 : p*N]
    s_rx[p, :] .= pulse .+ noise
end

# ========== 3. 脉冲压缩 ==========
println("[3/6] 脉冲压缩...")
t0 = time()

mf = Base.conj(Base.reverse(s_tx))
mf_ins = Insight.from_data(mf)

pc = Base.zeros(Complex{Float64}, N_PULSES, N)
for p in 1:N_PULSES
    pulse_ins = Insight.from_data(s_rx[p, :])
    conv = Insight.signal.fftconvolve(pulse_ins, mf_ins, mode="full")
    conv_jl = Insight.to_data(conv)
    full_len = Base.length(conv_jl)
    start = Base.div(full_len - N, 2) + 1
    pc[p, :] .= conv_jl[start:start+N-1]
end

println("  耗时: $(Base.round(time() - t0, digits=2)) 秒")

# ========== 4. 多普勒 FFT ==========
println("[4/6] 多普勒处理...")
t0 = time()

# Column-by-column FFT using Insight (Julia 列优先，逐列处理)
doppler_fft_jl = Base.zeros(Complex{Float64}, N_PULSES, N)
for col in 1:N
    col_ins = Insight.from_data(pc[:, col])
    fft_col = Insight.fft(col_ins, n=Int64(N_PULSES))
    shifted = Insight.fftshift(fft_col)
    doppler_fft_jl[:, col] .= Insight.to_data(shifted)
end

println("  耗时: $(Base.round(time() - t0, digits=2)) 秒")

# ========== 5. CA-CFAR 检测 ==========
println("[5/6] CFAR 目标检测...")
t0 = time()

energy = Base.abs.(doppler_fft_jl)
# 直接传入 CFAR（与 C++ 相同的 (N_PULSES, N) 形状）
energy_ins = Insight.from_data(energy)

threshold, detections = Insight.signal.ca_cfar(energy_ins, Int32[2, 2], Int32[4, 4], 1e-5)
det_data = Insight.to_data(detections)
det_jl = Base.reshape(det_data .!= 0, N_PULSES, N)

println("  耗时: $(Base.round(time() - t0, digits=2)) 秒")

# ========== 6. 目标聚类 ==========
println("[6/6] 聚类目标...")

target_indices = findall(det_jl)
raw_count = Base.length(target_indices)

visited = Base.falses(raw_count)
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
        d = Base.sqrt((target_indices[i][1] - target_indices[j][1])^2 +
                 (target_indices[i][2] - target_indices[j][2])^2)
        if d <= 5.0
            visited[j] = true
            Base.push!(cluster, target_indices[j])
        end
    end
    center_d = Base.round(Int, sum(c[1] for c in cluster) / Base.length(cluster))
    center_r = Base.round(Int, sum(c[2] for c in cluster) / Base.length(cluster))
    Base.push!(targets, (center_d, center_r))
end

# ========== 输出 ==========
freqs_ins = Insight.fftfreq(Int64(N_PULSES), T_PRF)
doppler_bins_jl = Insight.to_data(Insight.fftshift(freqs_ins))

println("\n[检测结果]")
println("  原始检测点数: $raw_count, 聚类后: $(Base.length(targets))")

for (d_idx, r_idx) in targets
    range_m = (r_idx - 1 - PC_OFFSET) * RANGE_PER_BIN  # Julia 1-based → 0-based
    doppler_hz = doppler_bins_jl[d_idx]
    println("    → 距离: $(Base.lpad(Base.round(range_m, digits=2), 8)) 米, " *
            "多普勒: $(Base.lpad(Base.round(doppler_hz, digits=1), 8)) Hz")
end

# GPU — silent skip when not available
if Insight.load_backend("cuda")
    println("\n" * "=" ^ 60)
    println("  GPU 信息")
    println("=" ^ 60)
    dev = Insight.device_name(0)
    if !isempty(dev)
        println("  设备: $dev")
    end
end

println("\n完成！")
