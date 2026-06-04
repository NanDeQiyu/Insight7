-- demos/radar_task1.lua
-- 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 Lua 完整版)
-- 与 C++ / Python / Julia 版本算法完全对齐：使用 Insight FFT 处理复数信号

local ins = require("insight")
-- Auto-init with smart discovery (CPU + first GPU if available)

-- ========== 参数 (与 C++ 完全一致) ==========
local FS = 10e6
local T_PRF = 1e-4
local N = 1000
local B = 5e6
local TAU = 50e-6
local N_PULSES = 128
local SNR_DB = 10
local TARGET_DELAYS = { 30e-6, 45e-6 }
local TARGET_DOPPLERS = { 2000.0, -1500.0 }
local RANGE_RES = 3e8 / (2 * B)
local RANGE_PER_BIN = 3e8 / (2 * FS)
local PC_OFFSET = math.floor((N - 1) / 2)

print("=" .. string.rep("=", 59))
print("  雷达目标检测与多普勒分析 (Insight7 Lua)")
print("=" .. string.rep("=", 59))

print("\n[配置信息]")
print(string.format("  采样率: %.1f MHz, 带宽: %.1f MHz", FS / 1e6, B / 1e6))
print(string.format("  单脉冲采样点数: %d, 脉冲串数量: %d", N, N_PULSES))
print(string.format("  距离分辨率: %.2f 米, 距离门: %.2f 米", RANGE_RES, RANGE_PER_BIN))

-- ========== RNG: 使用 Insight 原生 RNG 确保跨语言一致 ==========
ins.seed(42)

-- ========== 1. LFM 发射信号 ==========
print("\n[1/6] 生成 LFM 信号...")
local t_arr = {}
local s_tx_r, s_tx_i = {}, {}
local sig_power = 0
for i = 1, N do
  local ti = (i - 1) / FS
  t_arr[i] = ti
  if ti < TAU then
    local phase = math.pi * (B / TAU) * ti * ti
    s_tx_r[i] = math.cos(phase)
    s_tx_i[i] = math.sin(phase)
  else
    s_tx_r[i] = 0.0
    s_tx_i[i] = 0.0
  end
  sig_power = sig_power + s_tx_r[i] ^ 2 + s_tx_i[i] ^ 2
end
sig_power = sig_power / N
local noise_sigma = math.sqrt(sig_power / 10 ^ (SNR_DB / 10) / 2)

-- ========== 2. 模拟回波 ==========
print("[2/6] 模拟回波信号...")
-- 生成噪声数组 (Insight 原生 RNG，跨语言一致)
local noise_r = ins.randn({ N_PULSES, N }, ins.float64) * noise_sigma
local noise_i = ins.randn({ N_PULSES, N }, ins.float64) * noise_sigma
local s_rx_r = {}
local s_rx_i = {}

for p = 1, N_PULSES do
  local slow_time = (p - 1) * T_PRF
  local pr, pi_ = {}, {}
  for i = 1, N do
    pr[i] = 0.0
    pi_[i] = 0.0
  end

  for tgt = 1, #TARGET_DELAYS do
    local ds = math.floor(TARGET_DELAYS[tgt] * FS)
    local doppler = TARGET_DOPPLERS[tgt]
    local tr, ti_ = {}, {}
    for i = 1, N do
      tr[i] = 0.0
      ti_[i] = 0.0
    end
    if ds < N then
      for i = ds + 1, N do
        tr[i] = s_tx_r[i - ds]
        ti_[i] = s_tx_i[i - ds]
      end
    end
    for i = 1, N do
      local phase = 2 * math.pi * doppler * t_arr[i]
      local c, s = math.cos(phase), math.sin(phase)
      local nr = tr[i] * c - ti_[i] * s
      local ni = tr[i] * s + ti_[i] * c
      tr[i], ti_[i] = nr, ni
    end
    local sph = 2 * math.pi * doppler * slow_time
    local sc, ss = math.cos(sph), math.sin(sph)
    for i = 1, N do
      local nr = tr[i] * sc - ti_[i] * ss
      local ni = tr[i] * ss + ti_[i] * sc
      tr[i], ti_[i] = nr, ni
    end
    for i = 1, N do
      pr[i] = pr[i] + tr[i]
      pi_[i] = pi_[i] + ti_[i]
    end
  end

  for i = 1, N do
    pr[i] = pr[i] + noise_r:get((p - 1) * N + (i - 1))
    pi_[i] = pi_[i] + noise_i:get((p - 1) * N + (i - 1))
  end
  s_rx_r[p] = pr
  s_rx_i[p] = pi_
end

-- ========== 3. 脉冲压缩 (使用复数 fftconvolve) ==========
print("[3/6] 脉冲压缩...")
local t0 = os.clock()

-- 匹配滤波器 (复数共轭反转)
local mf_r, mf_i = {}, {}
for i = 1, N do
  mf_r[i] = s_tx_r[N + 1 - i]
  mf_i[i] = -s_tx_i[N + 1 - i]
end
local mf_c = ins.to_complex(ins.from_table(mf_r), ins.from_table(mf_i))

local full_len = 2 * N - 1
local start = math.floor((full_len - N) / 2)

local pc_r = {}
local pc_i = {}

for p = 1, N_PULSES do
  local pulse_c = ins.to_complex(ins.from_table(s_rx_r[p]), ins.from_table(s_rx_i[p]))
  local conv = ins.signal.fftconvolve(pulse_c, mf_c, "full")
  local conv_r = ins.real(conv)
  local conv_i = ins.imag(conv)

  pc_r[p] = {}
  pc_i[p] = {}
  for i = 1, N do
    pc_r[p][i] = conv_r:get(start + (i - 1))
    pc_i[p][i] = conv_i:get(start + (i - 1))
  end
end

print(string.format("  耗时: %.2f 秒", os.clock() - t0))

-- ========== 4. 多普勒 FFT (使用 Insight 批量 FFT) ==========
print("[4/6] 多普勒处理...")
t0 = os.clock()

-- 构建 2D 复数数组 [N_PULSES, N]
local pc_r_2d = {}
local pc_i_2d = {}
for p = 1, N_PULSES do
  pc_r_2d[p] = pc_r[p]
  pc_i_2d[p] = pc_i[p]
end
local pc_c = ins.to_complex(ins.from_table(pc_r_2d), ins.from_table(pc_i_2d))

-- 沿 axis 0 (脉冲维) 做 FFT，然后 fftshift
local doppler_fft = ins.fftshift(ins.fft(pc_c, N_PULSES, 0), 0)

-- 提取实部和虚部，计算能量
local dr = ins.real(doppler_fft)
local di = ins.imag(doppler_fft)
local energy_arr = ins.sqrt(ins.add(ins.mul(dr, dr), ins.mul(di, di)))

print(string.format("  耗时: %.2f 秒", os.clock() - t0))

-- ========== 5. CA-CFAR 检测 ==========
print("[5/6] CFAR 目标检测...")
t0 = os.clock()

local cfar_result = ins.signal.ca_cfar(energy_arr, { 2, 2 }, { 4, 4 }, 1e-5)
local detections = cfar_result[2]

local det = {}
for idx = 0, detections.numel - 1 do
  det[idx + 1] = detections:get(idx) ~= 0
end

print(string.format("  耗时: %.2f 秒", os.clock() - t0))

-- ========== 6. 目标聚类 ==========
print("[6/6] 聚类目标...")
local target_indices = {}
for idx = 1, N_PULSES * N do
  if det[idx] then
    local d = math.floor((idx - 1) / N)
    local r = (idx - 1) % N
    target_indices[#target_indices + 1] = { d, r }
  end
end

local raw_count = #target_indices
local visited = {}
for i = 1, raw_count do
  visited[i] = false
end

local targets = {}
for i = 1, raw_count do
  if not visited[i] then
    visited[i] = true
    local sum_d = target_indices[i][1]
    local sum_r = target_indices[i][2]
    local count = 1
    for j = i + 1, raw_count do
      if not visited[j] then
        local dist = math.sqrt(
          (target_indices[i][1] - target_indices[j][1]) ^ 2 + (target_indices[i][2] - target_indices[j][2]) ^ 2
        )
        if dist <= 5.0 then
          visited[j] = true
          sum_d = sum_d + target_indices[j][1]
          sum_r = sum_r + target_indices[j][2]
          count = count + 1
        end
      end
    end
    local center_d = math.floor(sum_d / count + 0.5)
    local center_r = math.floor(sum_r / count + 0.5)
    targets[#targets + 1] = { center_d, center_r }
  end
end

-- ========== 输出 ==========
local doppler_bins = {}
for i = 1, N_PULSES do
  doppler_bins[i] = ((i - 1) - math.floor(N_PULSES / 2)) * (1.0 / (N_PULSES * T_PRF))
end

print("\n[检测结果]")
print(string.format("  原始检测点数: %d, 聚类后: %d", raw_count, #targets))

for _, tgt in ipairs(targets) do
  local d_idx, r_idx = tgt[1], tgt[2]
  local range_m = (r_idx - PC_OFFSET) * RANGE_PER_BIN
  local doppler_hz = doppler_bins[d_idx + 1] or 0
  print(string.format("    → 距离: %7.2f 米, 多普勒: %8.1f Hz", range_m, doppler_hz))
end

-- GPU — silent skip when not available
local has_gpu = false
do
  local ok, _ = pcall(function()
    ins.load_backend("cuda")
    -- Verify GPU actually works by trying a device transfer
    local test = ins.from_table({ 1.0 }):to(ins.GPUPlace(0))
    test:to(ins.CPUPlace())
  end)
  has_gpu = ok
end

if has_gpu then
  print("\n============================================================")
  print("  GPU 信息")
  print("============================================================")
  local dev = ins.device_name("gpu")
  if dev and dev ~= "" then
    print(string.format("  设备: %s", dev))
  end
end

print("\n完成！")
