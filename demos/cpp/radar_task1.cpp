// demos/radar_task1.cpp
// 雷达目标检测与多普勒分析 — 比赛任务1
// 用 Insight7 C++ API 复刻 numpy+scipy 的完整处理流程
#include "insight/insight.h"
#include "insight/ops/fft.h"
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
#endif
#include "insight/ops/random.h"
#include "insight/ops/signal.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

static void separator(const char *title) {
  printf("\n============================================================\n");
  printf("  %s\n", title);
  printf("============================================================\n");
}

static bool gpu_available() {
  try {
    set_device(GPUPlace(0));
    return true;
  } catch (...) {
    return false;
  }
}

// RNG: use Insight's native seed + randn for cross-language alignment
static void init_rng() { ins::seed(42); }
static const double *noise_r_data = nullptr;
static const double *noise_i_data = nullptr;

// ========== 参数 ==========
static constexpr double fs = 10e6;   // 采样率
static constexpr double T = 1e-4;    // 脉冲重复周期
static constexpr int N = 1000;       // 单脉冲采样点数
static constexpr double B = 5e6;     // 带宽
static constexpr double tau = 50e-6; // 脉宽
static constexpr int n_pulses = 128;
static constexpr double SNR_dB = 10;
static constexpr double target_delays[] = {30e-6, 45e-6};
static constexpr double target_dopplers[] = {2000.0, -1500.0};
static constexpr int n_targets = 2;
static constexpr double range_res = 3e8 / (2 * B);      // 30m
static constexpr double range_per_bin = 3e8 / (2 * fs); // 15m
static constexpr double max_range = range_per_bin * N;  // 15000m
static constexpr double max_doppler = 1.0 / T;          // 10000Hz
static constexpr int pc_offset = (N - 1) / 2; // pulse compression crop offset

struct Task1Result {
  std::vector<std::pair<int, int>> targets;
  std::vector<double> energy;
  double pc_ms, doppler_ms, cfar_ms, total_ms;
  const char *device;
};

static Task1Result run_task1(Place place) {
  const char *dev_str = (place.kind() == DeviceKind::GPU) ? "GPU" : "CPU";
  auto t_start = std::chrono::high_resolution_clock::now();

  // ========== 1. 生成 LFM 发射信号 ==========
  std::vector<std::complex<double>> s_tx(N);
  for (int i = 0; i < N; ++i) {
    double t_i = i / fs;
    if (t_i < tau) {
      double phase = M_PI * (B / tau) * t_i * t_i;
      s_tx[i] = std::exp(std::complex<double>(0.0, phase));
    } else {
      s_tx[i] = 0.0;
    }
  }

  double sig_power = 0;
  for (auto &v : s_tx)
    sig_power += std::norm(v);
  sig_power /= N;
  double noise_sigma =
      std::sqrt(sig_power / std::pow(10.0, SNR_dB / 10.0) / 2.0);

  std::vector<double> t_vec(N);
  for (int i = 0; i < N; ++i)
    t_vec[i] = i / fs;

  // ========== 2. 模拟回波 (逐目标施加 Doppler) ==========
  // 使用 Insight 原生 RNG 确保跨语言一致
  init_rng();
  Array noise_r_arr =
      randn(Shape({n_pulses, N}), DType::F64, CPUPlace()) * noise_sigma;
  Array noise_i_arr =
      randn(Shape({n_pulses, N}), DType::F64, CPUPlace()) * noise_sigma;
  noise_r_data = noise_r_arr.data<double>();
  noise_i_data = noise_i_arr.data<double>();

  std::vector<std::complex<double>> s_rx_flat(n_pulses * N);

  for (int p = 0; p < n_pulses; ++p) {
    double slow_time = p * T;
    std::vector<std::complex<double>> pulse(N, 0.0);

    // 逐目标：施加延迟 + 快时间 Doppler
    for (int tgt = 0; tgt < n_targets; ++tgt) {
      int delay_samp = (int)(target_delays[tgt] * fs);
      double doppler = target_dopplers[tgt];

      // 临时存放该目标的回波
      std::vector<std::complex<double>> tgt_pulse(N, 0.0);
      if (delay_samp < N) {
        for (int i = delay_samp; i < N; ++i)
          tgt_pulse[i] = s_tx[i - delay_samp];
      }

      // 快时间维 Doppler
      for (int i = 0; i < N; ++i) {
        double phase = 2.0 * M_PI * doppler * t_vec[i];
        tgt_pulse[i] *= std::exp(std::complex<double>(0.0, phase));
      }

      // 慢时间维 Doppler
      {
        double phase = 2.0 * M_PI * doppler * slow_time;
        auto slow_ph = std::exp(std::complex<double>(0.0, phase));
        for (int i = 0; i < N; ++i)
          tgt_pulse[i] *= slow_ph;
      }

      // 累加到总回波
      for (int i = 0; i < N; ++i)
        pulse[i] += tgt_pulse[i];
    }

    // 加噪声 (使用 Insight 原生 RNG)
    for (int i = 0; i < N; ++i) {
      std::complex<double> noise(noise_r_data[p * N + i],
                                 noise_i_data[p * N + i]);
      s_rx_flat[p * N + i] = pulse[i] + noise;
    }
  }

  // ========== 3. 脉冲压缩 ==========
  auto t_pc0 = std::chrono::high_resolution_clock::now();

  // Matched filter (create once)
  std::vector<std::complex<double>> mf(N);
  for (int i = 0; i < N; ++i)
    mf[i] = std::conj(s_tx[N - 1 - i]);
  Array mf_arr = to_array(mf, DType::C64, CPUPlace());
  if (place.kind() == DeviceKind::GPU)
    mf_arr = mf_arr.to(place);

  // Pulse compression (reuse output buffer for FFTW plan cache)
  std::vector<std::complex<double>> pc_flat(n_pulses * N);
  Array pulse_arr = zeros(Shape({N}), DType::C64, CPUPlace());
  Array conv_out;

  for (int p = 0; p < n_pulses; ++p) {
    std::vector<std::complex<double>> pulse_vec(
        s_rx_flat.begin() + p * N, s_rx_flat.begin() + (p + 1) * N);

    // Copy into reusable buffer
    if (place.kind() == DeviceKind::GPU) {
      Array tmp = to_array(pulse_vec, DType::C64, CPUPlace());
      pulse_arr = tmp.to(place);
    } else {
      std::memcpy(pulse_arr.data<char>(), pulse_vec.data(), N * 16);
    }

    Array conv = signal::fftconvolve(pulse_arr, mf_arr, "full");
    Array conv_cpu =
        (place.kind() == DeviceKind::GPU) ? conv.to(CPUPlace()) : conv;
    const std::complex<double> *cd =
        reinterpret_cast<const std::complex<double> *>(conv_cpu.data<char>());
    int start = (2 * N - 1 - N) / 2;
    for (int i = 0; i < N; ++i)
      pc_flat[p * N + i] = cd[start + i];
  }

  auto t_pc1 = std::chrono::high_resolution_clock::now();
  double pc_ms =
      std::chrono::duration<double, std::milli>(t_pc1 - t_pc0).count();

  // ========== 4. 多普勒处理 (2D FFT, axis=0) ==========
  auto t_dp0 = std::chrono::high_resolution_clock::now();

  Array pc_arr =
      to_array(pc_flat, Shape({n_pulses, N}), DType::C64, CPUPlace());
  if (place.kind() == DeviceKind::GPU)
    pc_arr = pc_arr.to(place);

  Array doppler_arr = fft::fftshift(fft::fft(pc_arr, n_pulses, 0), 0);
  if (place.kind() == DeviceKind::GPU)
    doppler_arr = doppler_arr.to(CPUPlace());

  auto t_dp1 = std::chrono::high_resolution_clock::now();
  double doppler_ms =
      std::chrono::duration<double, std::milli>(t_dp1 - t_dp0).count();

  // ========== 5. CA-CFAR ==========
  auto t_cf0 = std::chrono::high_resolution_clock::now();

  Array energy_arr = abs(doppler_arr);
  auto [threshold, detections] =
      signal::ca_cfar(energy_arr, {2, 2}, {4, 4}, 1e-5);

  auto t_cf1 = std::chrono::high_resolution_clock::now();
  double cfar_ms =
      std::chrono::duration<double, std::milli>(t_cf1 - t_cf0).count();

  // ========== 6. 目标聚类 ==========
  const bool *det = detections.data<bool>();
  std::vector<std::pair<int, int>> raw_det;
  for (int d = 0; d < n_pulses; ++d)
    for (int r = 0; r < N; ++r)
      if (det[d * N + r])
        raw_det.push_back({d, r});

  std::vector<bool> visited(raw_det.size(), false);
  std::vector<std::pair<int, int>> targets;
  for (size_t i = 0; i < raw_det.size(); ++i) {
    if (visited[i])
      continue;
    visited[i] = true;
    double sum_d = raw_det[i].first, sum_r = raw_det[i].second;
    int count = 1;
    for (size_t j = i + 1; j < raw_det.size(); ++j) {
      if (visited[j])
        continue;
      double dist =
          std::sqrt(std::pow(raw_det[i].first - raw_det[j].first, 2) +
                    std::pow(raw_det[i].second - raw_det[j].second, 2));
      if (dist <= 5.0) {
        visited[j] = true;
        sum_d += raw_det[j].first;
        sum_r += raw_det[j].second;
        count++;
      }
    }
    targets.push_back(
        {(int)std::round(sum_d / count), (int)std::round(sum_r / count)});
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  Task1Result result;
  result.targets = targets;
  // Extract energy data from Insight Array for plotting
  {
    const double *e_data = energy_arr.data<double>();
    result.energy.assign(e_data, e_data + n_pulses * N);
  }
  result.pc_ms = pc_ms;
  result.doppler_ms = doppler_ms;
  result.cfar_ms = cfar_ms;
  result.total_ms = total_ms;
  result.device = dev_str;
  return result;
}

// ========== 绘图 ==========
#ifdef INSIGHT_USE_MATPLOT
static void save_plots(const Task1Result &result, const char *prefix) {
  separator("绘图");

  // Pre-compute data (shared between plots)
  std::vector<double> energy_db(n_pulses * N);
  for (int i = 0; i < n_pulses * N; ++i)
    energy_db[i] = 20.0 * std::log10(result.energy[i] + 1e-8);
  Array energy_arr =
      to_array(energy_db, Shape({n_pulses, N}), DType::F64, CPUPlace());

  std::vector<double> range_axis(N), doppler_axis(n_pulses);
  for (int i = 0; i < N; ++i)
    range_axis[i] = (i - pc_offset) * range_per_bin;
  for (int i = 0; i < n_pulses; ++i)
    doppler_axis[i] = (i - n_pulses / 2) * (1.0 / (n_pulses * T));
  Array doppler_arr = to_array(doppler_axis, DType::F64, CPUPlace());

  int max_r = 0;
  double max_e = 0;
  for (int r = 0; r < N; ++r) {
    double e = 0;
    for (int d = 0; d < n_pulses; ++d)
      e = std::max(e, result.energy[d * N + r]);
    if (e > max_e) {
      max_e = e;
      max_r = r;
    }
  }
  std::vector<double> doppler_slice(n_pulses);
  for (int d = 0; d < n_pulses; ++d)
    doppler_slice[d] = result.energy[d * N + max_r];
  Array ds_arr = to_array(doppler_slice, DType::F64, CPUPlace());

  // Plots rendered directly — gnuplot stdout → /dev/null (gnuplot.cpp patch).
  std::string p1 = std::string(prefix) + "_range_doppler.png";
  std::string p2 = std::string(prefix) + "_doppler_slice.png";
  plot::figure();
  plot::save(p1);
  plot::imshow(energy_arr);
  plot::colorbar();
  plot::title("Range-Doppler Map");
  plot::xlabel("Range Bin");
  plot::ylabel("Doppler Bin");
  plot::figure();
  plot::save(p2);
  plot::plot(doppler_arr, ds_arr);
  plot::title("Doppler Spectrum (max range bin)");
  plot::xlabel("Doppler Frequency [Hz]");
  plot::ylabel("Amplitude");
  plot::grid(true);
  printf("  已保存: %s, %s\n", p1.c_str(), p2.c_str());
}
#endif // INSIGHT_USE_MATPLOT

int main() {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);
#ifdef SIGPIPE
  // Ignore SIGPIPE to prevent process crash when gnuplot is unavailable.
  // matplotplusplus spawns gnuplot via popen; if gnuplot is not installed
  // the pipe breaks and SIGPIPE would kill the process (not catchable by
  // try/catch since it is a signal, not a C++ exception).
  std::signal(SIGPIPE, SIG_IGN);
#endif

  ins::init({"cpu"});
  separator("比赛任务1：雷达目标检测与多普勒分析");

  printf("\n[配置信息]\n");
  printf("  采样率: %.1f MHz\n", fs / 1e6);
  printf("  脉冲宽度: %.1f us\n", tau * 1e6);
  printf("  信号带宽: %.1f MHz\n", B / 1e6);
  printf("  单脉冲采样点数: %d\n", N);
  printf("  脉冲串数量: %d\n", n_pulses);
  printf("  距离分辨率: %.2f 米\n", range_res);
  printf("  最大不模糊距离: %.2f 米\n", max_range);
  printf("  最大不模糊多普勒: %.1f Hz\n", max_doppler);

  // CPU
  separator("CPU 运行");
  set_device(CPUPlace());
  Task1Result cpu = run_task1(CPUPlace());
  printf("  CPU 耗时: %.2f ms (PC: %.1f, Doppler: %.1f, CFAR: %.1f)\n",
         cpu.total_ms, cpu.pc_ms, cpu.doppler_ms, cpu.cfar_ms);
  printf("  聚类后目标数: %zu\n", cpu.targets.size());

  std::vector<double> doppler_bins(n_pulses);
  for (int i = 0; i < n_pulses; ++i)
    doppler_bins[i] = (i - n_pulses / 2) * (1.0 / (n_pulses * T));

  printf("\n[CPU 检测结果]\n");
  for (auto &[d, r] : cpu.targets) {
    double range_m = (r - pc_offset) * range_per_bin;
    printf("  → 距离: %7.2f 米, 多普勒: %8.1f Hz\n", range_m, doppler_bins[d]);
  }

#ifdef INSIGHT_USE_MATPLOT
  if (system("gnuplot --version > /dev/null 2>&1") == 0) {
    // Suppress SIGSEGV during plotting — matplotplusplus/cairo may crash
    // in headless environments. Main computation is already complete.
    // TODO: use platform-specific process isolation for Windows support.
    auto prev_segv = std::signal(SIGSEGV, [](int) { _exit(0); });
    try {
      save_plots(cpu, "task1_cpu");
      printf("  已保存: task1_cpu_*.png\n");
    } catch (const std::exception &e) {
      printf("[Warning] CPU plotting failed: %s\n", e.what());
    } catch (...) {
      printf("[Warning] CPU plotting failed (unknown error)\n");
    }
    std::signal(SIGSEGV, prev_segv);
  }
#endif

  printf("\n完成！\n");
  return 0;
}
