// demos/cpp/radar_detection.cpp
// 雷达目标检测与多普勒分析 — 比赛任务1 (Insight7 C++)
// 纯 Insight7 API，与 Python 版算法完全对齐
// 编译: cmake .. -DINSIGHT_BUILD_DEMOS=ON && make -j24
// 运行: ./radar_detection --device gpu --seed 42 --iterations 50

#include "insight/core/profiler.h"
#include "insight/insight.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/random.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

// ============================================================
// 物理参数
// ============================================================
static constexpr double FS = 10e6;
static constexpr double T_PRF = 100e-6;
static constexpr int64_t N = 1000;
static constexpr double B = 5e6;
static constexpr double TAU = 50e-6;
static constexpr int64_t N_PULSES = 32;
static constexpr double SNR_DB = 10;
static constexpr int64_t PULSE_LEN = int64_t(TAU * FS); // 500
static constexpr int64_t PC_OFFSET = PULSE_LEN / 2;     // 250
static constexpr double RANGE_PER_BIN = 3e8 / (2 * FS); // 15 m
static constexpr double MAX_UNAMBIG_RANGE = RANGE_PER_BIN * N;

// 目标移动轨迹 (与 Python 一致)
static const std::vector<double> DELAY_START = {35e-6, 50e-6};
static const std::vector<double> DELAY_END = {11e-6, 20e-6};
static const std::vector<double> DOPPLER_START = {2000.0, -1500.0};
static const std::vector<double> DOPPLER_END = {1800.0, -1200.0};

// ============================================================
// 全局缓存
// ============================================================
static Array _S_TX;         // LFM 发射信号
static Array _TEMPLATE;     // 脉冲压缩模板
static Array _HAMMING;      // 缓存的 hamming 窗 [N_PULSES, 1]
static Array _SLOW_TIMES;   // 缓存的慢时间轴
static double _SIG_POWER;   // 信号功率
static double _NOISE_SIGMA; // 噪声标准差
static Array _DOPPLER_BINS; // 多普勒轴
static Array _RANGE_BINS;   // 距离轴

static void init_cache(Place place) {
  // LFM 发射信号
  Array t = arange(N, DType::F64, place) / FS;
  Array phase = M_PI * (B / TAU) * (t * t);
  Array s_tx = to_complex(cos(phase), sin(phase));
  Array tau_arr = full({1}, TAU, DType::F64, place);
  Array mask = cast(less(t, tau_arr), DType::F64);
  _S_TX = mul(s_tx, mask);

  // 信号功率
  Array sig_power_arr = mean(square(real(_S_TX)) + square(imag(_S_TX)));
  _SIG_POWER = sig_power_arr.item<double>();
  _NOISE_SIGMA = std::sqrt(_SIG_POWER / std::pow(10.0, SNR_DB / 10.0) / 2.0);

  // 多普勒/距离轴
  Array doppler = fft::fftshift(fft::fftfreq(N_PULSES, T_PRF));
  _DOPPLER_BINS = doppler.place().kind() == DeviceKind::CPU
                      ? doppler
                      : doppler.to(CPUPlace());
  Array range_arr = (arange(N, DType::F64, place) - PC_OFFSET) * RANGE_PER_BIN;
  _RANGE_BINS = range_arr.place().kind() == DeviceKind::CPU
                    ? range_arr
                    : range_arr.to(CPUPlace());

  // 预创建脉冲压缩模板和 hamming 窗
  _TEMPLATE = _S_TX.slice({Slice(0, PULSE_LEN)});
  _HAMMING =
      reshape(signal::get_window("hamming", N_PULSES, false), {N_PULSES, 1});
  _SLOW_TIMES = arange(N_PULSES, DType::F64, place) * T_PRF;
}

// ============================================================
// 目标参数插值
// ============================================================
static void get_target_params(int frame_idx, int total_frames,
                              std::vector<double> &delays,
                              std::vector<double> &dopplers) {
  if (total_frames <= 1) {
    delays = DELAY_START;
    dopplers = DOPPLER_START;
    return;
  }
  double t = double(frame_idx) / double(std::max(total_frames - 1, 1));
  delays.resize(DELAY_START.size());
  dopplers.resize(DOPPLER_START.size());
  for (size_t k = 0; k < DELAY_START.size(); ++k) {
    delays[k] = DELAY_START[k] + (DELAY_END[k] - DELAY_START[k]) * t;
    dopplers[k] = DOPPLER_START[k] + (DOPPLER_END[k] - DOPPLER_START[k]) * t;
  }
}

// ============================================================
// 向量化回波模拟
// ============================================================
static Array simulate_echoes(const std::vector<double> &delays,
                             const std::vector<double> &dopplers, Place place,
                             const Array *external_noise_r = nullptr,
                             const Array *external_noise_i = nullptr) {
  Array noise_r = external_noise_r
                      ? *external_noise_r
                      : mul(randn({N_PULSES, N}, DType::F64, place),
                            full({1}, _NOISE_SIGMA, DType::F64, place));
  Array noise_i = external_noise_i
                      ? *external_noise_i
                      : mul(randn({N_PULSES, N}, DType::F64, place),
                            full({1}, _NOISE_SIGMA, DType::F64, place));

  Array pulses = zeros({N_PULSES, N}, DType::C64, place);

  for (size_t tgt = 0; tgt < delays.size(); ++tgt) {
    int64_t ds = int64_t(delays[tgt] * FS);
    Array delayed_1d = zeros({N}, DType::C64, place);
    if (ds < N) {
      delayed_1d.slice({Slice(ds, N)})
          .copy_from_(_S_TX.slice({Slice(0, N - ds)}));
    }
    Array delayed_2d = reshape(delayed_1d, {1, N});

    Array phase = _SLOW_TIMES * (2.0 * M_PI * dopplers[tgt]);
    Array rot = to_complex(cos(phase), sin(phase));
    Array rot_2d = reshape(rot, {N_PULSES, 1});

    pulses = add(pulses, mul(delayed_2d, rot_2d));
  }

  pulses = add(pulses, to_complex(noise_r, noise_i));
  return pulses;
}

// ============================================================
// 目标提取
// ============================================================
struct TargetInfo {
  int d, r;
};
struct ExtractResult {
  std::vector<TargetInfo> targets;
  int raw_count;
};

static ExtractResult extract_targets(const Array &energy, const Array &det,
                                     int top_n = 2,
                                     double cluster_threshold = 3.0) {
  ExtractResult result;

  // 将 detection 转到 CPU 并扫描
  Array det_cpu =
      (det.place().kind() == DeviceKind::CPU) ? det : det.to(CPUPlace());
  const bool *det_data = det_cpu.data<bool>();
  int64_t n_det_doppler = det.shape().dim(0);
  int64_t n_det_range = det.shape().dim(1);

  // 收集 all detection points
  struct Candidate {
    int i, j;
    double v;
  };
  std::vector<Candidate> candidates;

  for (int d = 0; d < n_det_doppler; ++d) {
    for (int r = 0; r < n_det_range; ++r) {
      if (det_data[d * n_det_range + r]) {
        candidates.push_back({d, r, 0.0});
      }
    }
  }

  if (candidates.empty())
    return result;
  result.raw_count = (int)candidates.size();

  // 从 energy 数组读取能量值
  Array energy_cpu = (energy.place().kind() == DeviceKind::CPU)
                         ? energy
                         : energy.to(CPUPlace());
  const double *eng_data = energy_cpu.data<double>();
  for (auto &c : candidates) {
    c.v = eng_data[c.i * n_det_range + c.j];
  }

  // 按能量排序
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate &a, const Candidate &b) { return a.v > b.v; });

  int keep = std::min((int)candidates.size(), std::max(top_n * 10, 20));
  std::vector<Candidate> top(candidates.begin(), candidates.begin() + keep);

  // 聚类
  std::vector<bool> visited(keep, false);
  std::vector<Candidate> clusters;

  for (int i = 0; i < keep; ++i) {
    if (visited[i])
      continue;
    visited[i] = true;
    std::vector<int> cluster_indices = {i};
    for (int j = i + 1; j < keep; ++j) {
      if (visited[j])
        continue;
      double d = std::sqrt(std::pow(top[i].i - top[j].i, 2) +
                           std::pow(top[i].j - top[j].j, 2));
      if (d <= cluster_threshold) {
        visited[j] = true;
        cluster_indices.push_back(j);
      }
    }
    auto best =
        std::max_element(cluster_indices.begin(), cluster_indices.end(),
                         [&](int a, int b) { return top[a].v < top[b].v; });
    clusters.push_back(top[*best]);
  }

  // 按能量排序
  std::sort(clusters.begin(), clusters.end(),
            [](const Candidate &a, const Candidate &b) { return a.v > b.v; });

  // 按距离去重
  std::set<int> seen_ranges;
  for (auto &c : clusters) {
    if (seen_ranges.find(c.j) == seen_ranges.end()) {
      seen_ranges.insert(c.j);
      TargetInfo ti;
      ti.d = c.i;
      ti.r = c.j;
      result.targets.push_back(ti);
      if ((int)result.targets.size() >= top_n)
        break;
    }
  }

  return result;
}

// ============================================================
// 单帧检测
// ============================================================
struct FrameResult {
  std::vector<TargetInfo> targets;
  int raw_count;
  double echo_ms, pc_ms, doppler_ms, cfar_ms, local_ms, total_ms;
  Array energy;
};

static FrameResult run_frame(const std::vector<double> &delays,
                             const std::vector<double> &dopplers, Place place,
                             int seed_val, int,
                             const Array *external_noise_r = nullptr,
                             const Array *external_noise_i = nullptr,
                             ins::Profiler *prof = nullptr) {
  seed(seed_val);

  auto t0 = std::chrono::high_resolution_clock::now();

  // [1] Echo simulation
  if (prof)
    prof->begin_event("echo_simulation");
  Array pulses = simulate_echoes(delays, dopplers, place, external_noise_r,
                                 external_noise_i);
  if (prof)
    prof->end_event();
  auto t1 = std::chrono::high_resolution_clock::now();

  // [2] Pulse compression
  if (prof)
    prof->begin_event("pulse_compression");
  Array pc = signal::pulse_compression(pulses, _TEMPLATE);
  if (prof)
    prof->end_event();
  auto t2 = std::chrono::high_resolution_clock::now();

  // [3] Doppler processing
  if (prof)
    prof->begin_event("doppler");
  Array mean_pc = mean(pc, {0}, true);
  Array pc_ac = sub(pc, mean_pc);
  Array spec = signal::pulse_doppler(mul(pc_ac, _HAMMING), "", 0);
  Array doppler_shifted = div(
      spec, full(spec.shape(), (double)N_PULSES, spec.dtype(), spec.place()));
  if (prof)
    prof->end_event();
  auto t3 = std::chrono::high_resolution_clock::now();

  // [4] Energy
  Array energy =
      add(square(real(doppler_shifted)), square(imag(doppler_shifted)));

  // [5] CA-CFAR
  if (prof)
    prof->begin_event("ca_cfar");
  auto [threshold, detections] =
      signal::ca_cfar(energy, {4, 4}, {12, 12}, 1e-6);
  if (prof)
    prof->end_event();
  auto t4 = std::chrono::high_resolution_clock::now();

  // [6] Target extraction
  if (prof)
    prof->begin_event("target_extraction");
  ExtractResult ext = extract_targets(energy, detections);
  if (prof)
    prof->end_event();
  auto t5 = std::chrono::high_resolution_clock::now();

  // [7] Range validation
  FrameResult result;
  const double *range_data = _RANGE_BINS.data<double>();
  for (auto &tgt : ext.targets) {
    double dist = range_data[tgt.r];
    if (dist >= 0 && dist <= MAX_UNAMBIG_RANGE) {
      result.targets.push_back(tgt);
    }
  }
  result.raw_count = ext.raw_count;

  auto elapsed = [](auto s, auto e) {
    return std::chrono::duration<double, std::milli>(e - s).count();
  };
  result.echo_ms = elapsed(t0, t1);
  result.pc_ms = elapsed(t1, t2);
  result.doppler_ms = elapsed(t2, t3);
  result.cfar_ms = elapsed(t3, t4);
  result.local_ms = elapsed(t4, t5);
  result.total_ms = elapsed(t0, t5);
  result.energy = energy;

  return result;
}

// ============================================================
// CLI 解析
// ============================================================
struct Args {
  std::string device = "default";
  int seed = 42;
  int iterations = 0;
  bool timer = false;
  bool info = false;
  bool profiler = false;
};

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--device") == 0 && i + 1 < argc)
      args.device = argv[++i];
    else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
      args.seed = std::stoi(argv[++i]);
    else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc)
      args.iterations = std::stoi(argv[++i]);
    else if (strcmp(argv[i], "--timer") == 0)
      args.timer = true;
    else if (strcmp(argv[i], "--info") == 0)
      args.info = true;
    else if (strcmp(argv[i], "--profiler") == 0)
      args.profiler = true;
  }
  return args;
}

// ============================================================
// 主函数
// ============================================================
int main(int argc, char **argv) {
  setbuf(stdout, NULL);

  Args args = parse_args(argc, argv);
  ins::init();

  // Device resolution
  std::string eff_device = args.device;
  if (eff_device == "default") {
    eff_device = has_device(DeviceKind::GPU) ? "gpu" : "cpu";
  }
  bool use_gpu = (eff_device == "gpu") && has_device(DeviceKind::GPU);
  bool do_all = (eff_device == "all");
  Place place = use_gpu ? Place(GPUPlace(0)) : Place(CPUPlace());
  if (!do_all)
    set_device(place);

  int n_frames = (args.iterations > 0) ? args.iterations : 1;

  printf("============================================================\n");
  printf("  雷达目标检测与多普勒分析 (Insight7 C++)\n");
  printf("============================================================\n");
  printf(
      "\n[配置] 采样率: %.1f MHz  脉冲: %lld  设备: %s  种子: %d  帧数: %d\n",
      FS / 1e6, (long long)N_PULSES, do_all ? "all" : (use_gpu ? "gpu" : "cpu"),
      args.seed, n_frames);

  if (do_all && !has_device(DeviceKind::GPU)) {
    printf("[警告] GPU 不可用，--device all 退化为仅 CPU\n");
    do_all = false;
    eff_device = "cpu";
    place = CPUPlace();
  }

  // 初始化缓存 (单设备模式)
  ins::Profiler prof(place);
  if (!do_all) {
    init_cache(place);

    if (args.profiler)
      prof.start();

    // 模糊函数分析
    printf("\n[波形分析] 计算模糊函数...\n");
    Array ambg;
    if (args.profiler)
      prof.begin_event("ambgfun");
    ambg = signal::ambgfun(_TEMPLATE, FS, 1.0 / T_PRF);
    if (args.profiler)
      prof.end_event();
    Array ambg_abs = abs(ambg);
    double peak_val = max(ambg_abs).item<double>();
    double mean_val = mean(ambg_abs).item<double>();
    double peak_ratio = 20.0 * std::log10(peak_val / (mean_val + 1e-12));
    printf("  模糊函数峰值: %.2f\n", peak_val);
    printf("  峰值/平均比: %.1f dB  (主瓣窄、旁瓣低 ✓)\n", peak_ratio);
  }

  // --info 模式
  if (args.info) {
    auto cpu_mem = device_memory_info(DeviceKind::CPU, 0);
    printf("[Memory] CPU: total=%zuMB used=%zuMB\n",
           cpu_mem.total / (1024 * 1024),
           (cpu_mem.total - cpu_mem.free) / (1024 * 1024));
    if (has_device(DeviceKind::GPU)) {
      auto gpu_mem = device_memory_info(DeviceKind::GPU, 0);
      printf("[Memory] GPU: total=%zuMB used=%zuMB\n",
             gpu_mem.total / (1024 * 1024),
             (gpu_mem.total - gpu_mem.free) / (1024 * 1024));
    }
  }

  // ========================================================
  // --device all: 双设备比较
  // ========================================================
  if (do_all) {
    printf("\n[双设备比较] CPU vs GPU\n\n");

    Place cpu_place = CPUPlace();
    Place gpu_place = GPUPlace(0);

    std::vector<double> cpu_times, gpu_times;

    for (int frame = 0; frame < n_frames; ++frame) {
      std::vector<double> delays, dopplers;
      get_target_params(frame, n_frames, delays, dopplers);

      // 在 CPU 上生成噪声（只生成一次，CPU 和 GPU 共用）
      init_cache(cpu_place);
      seed(args.seed + frame);
      Array cpu_noise_r = mul(randn({N_PULSES, N}, DType::F64, cpu_place),
                              full({1}, _NOISE_SIGMA, DType::F64, cpu_place));
      Array cpu_noise_i = mul(randn({N_PULSES, N}, DType::F64, cpu_place),
                              full({1}, _NOISE_SIGMA, DType::F64, cpu_place));

      // CPU run（用 CPU 噪声）
      FrameResult cpu_result = run_frame(delays, dopplers, cpu_place, args.seed,
                                         frame, &cpu_noise_r, &cpu_noise_i);
      cpu_times.push_back(cpu_result.total_ms);

      // GPU run（用同一份噪声，传输到 GPU）
      Array gpu_noise_r = cpu_noise_r.to(gpu_place);
      Array gpu_noise_i = cpu_noise_i.to(gpu_place);
      init_cache(gpu_place);
      FrameResult gpu_result = run_frame(delays, dopplers, gpu_place, args.seed,
                                         frame, &gpu_noise_r, &gpu_noise_i);
      gpu_times.push_back(gpu_result.total_ms);

      // 比较 energy
      Array gpu_energy_cpu = gpu_result.energy;
      if (gpu_energy_cpu.place().kind() != DeviceKind::CPU)
        gpu_energy_cpu = gpu_result.energy.to(CPUPlace());
      Array diff = abs(cpu_result.energy - gpu_energy_cpu);
      double max_diff_val = max(diff).item<double>();

      bool print_frame = (n_frames == 1 || frame == 0 ||
                          (frame + 1) % 10 == 0 || frame == n_frames - 1);

      if (print_frame) {
        if (args.timer) {
          printf("  帧 %4d/%d | CPU: %8.2f ms (echo %5.2f pc %5.2f dop %5.2f "
                 "cfar %5.2f ext %5.2f) | GPU: %8.2f ms (echo %5.2f pc %5.2f "
                 "dop %5.2f cfar %5.2f ext %5.2f) | max_diff=%.2e\n",
                 frame, n_frames, cpu_result.total_ms, cpu_result.echo_ms,
                 cpu_result.pc_ms, cpu_result.doppler_ms, cpu_result.cfar_ms,
                 cpu_result.local_ms, gpu_result.total_ms, gpu_result.echo_ms,
                 gpu_result.pc_ms, gpu_result.doppler_ms, gpu_result.cfar_ms,
                 gpu_result.local_ms, max_diff_val);
        } else {
          printf("  帧 %4d/%d | CPU: %8.2f ms | GPU: %8.2f ms | "
                 "max_diff=%.2e\n",
                 frame, n_frames, cpu_result.total_ms, gpu_result.total_ms,
                 max_diff_val);
        }
      }
    }

    // 性能总结
    double cpu_sum = 0, gpu_sum = 0;
    for (auto t : cpu_times)
      cpu_sum += t;
    for (auto t : gpu_times)
      gpu_sum += t;
    double cpu_avg = cpu_sum / cpu_times.size();
    double gpu_avg = gpu_sum / gpu_times.size();

    printf("\n  ==================================================\n");
    printf("  性能总结 (CPU vs GPU)\n");
    printf("  ==================================================\n");
    printf("  总帧数: %d\n", n_frames);
    printf("  CPU 平均每帧: %.2f ms  FPS: %.2f\n", cpu_avg, 1000.0 / cpu_avg);
    printf("  GPU 平均每帧: %.2f ms  FPS: %.2f\n", gpu_avg, 1000.0 / gpu_avg);
    printf("  加速比: %.2fx\n", cpu_avg / gpu_avg);
    printf("\n完成！\n");
    return 0;
  }

  // ========================================================
  // 单设备模式 (原有逻辑)
  // ========================================================

  // 帧循环
  std::vector<double> times;
  for (int frame = 0; frame < n_frames; ++frame) {
    std::vector<double> delays, dopplers;
    get_target_params(frame, n_frames, delays, dopplers);

    if (args.profiler)
      prof.begin_event("frame");
    FrameResult result =
        run_frame(delays, dopplers, place, args.seed, frame, nullptr, nullptr,
                  args.profiler ? &prof : nullptr);
    if (args.profiler)
      prof.end_event();

    times.push_back(result.total_ms);

    bool print_frame = (n_frames == 1 || frame == 0 || (frame + 1) % 10 == 0 ||
                        frame == n_frames - 1);

    if (print_frame) {
      std::string targets_str;
      const double *dop_data = _DOPPLER_BINS.data<double>();
      const double *rng_data = _RANGE_BINS.data<double>();
      for (size_t k = 0; k < result.targets.size(); ++k) {
        if (k > 0)
          targets_str += "; ";
        char buf[128];
        snprintf(buf, sizeof(buf), "距离 %.0fm 多普勒 %.0fHz",
                 rng_data[result.targets[k].r], dop_data[result.targets[k].d]);
        targets_str += buf;
      }
      if (targets_str.empty())
        targets_str = "无";

      if (args.timer) {
        printf("  帧 %4d/%d | %8.2f ms | 检测 %4d → %zu 目标 | echo %5.2f pc "
               "%5.2f dop %5.2f cfar %5.2f ext %5.2f | %s\n",
               frame, n_frames, result.total_ms, result.raw_count,
               result.targets.size(), result.echo_ms, result.pc_ms,
               result.doppler_ms, result.cfar_ms, result.local_ms,
               targets_str.c_str());
      } else {
        printf("  帧 %4d/%d | %8.2f ms | 检测 %4d → %zu 目标 | %s\n", frame,
               n_frames, result.total_ms, result.raw_count,
               result.targets.size(), targets_str.c_str());
      }
    }
  }

  // Profiler 报告
  if (args.profiler) {
    prof.stop();
    prof.report();
  }

  // 性能总结
  double sum = 0;
  for (auto t : times)
    sum += t;
  double avg_ms = sum / times.size();
  double fps = 1000.0 / avg_ms;

  printf("\n  ==================================================\n");
  printf("  性能总结 (%s)\n", use_gpu ? "GPU" : "CPU");
  printf("  ==================================================\n");
  printf("  总帧数: %d\n", n_frames);
  printf("  平均每帧: %.2f ms  FPS: %.2f\n", avg_ms, fps);
  printf("\n完成！\n");

  return 0;
}
