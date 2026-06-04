// demos/fft_demo.cpp
// Demonstrates: FFT signal processing on CPU and GPU
#include "insight/insight.h"
#include "insight/ops/fft.h"
#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

static void separator(const char *title) {
  printf("\n========================================\n");
  printf("  %s\n", title);
  printf("========================================\n");
}

static bool gpu_available() {
  try {
    set_device(GPUPlace(0));
    return true;
  } catch (...) {
    return false;
  }
}

static void run_fft_cpu() {
  separator("CPU FFT");

  // Generate a signal: sum of two sinusoids
  int n = 64;
  std::vector<float> signal(n);
  for (int i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / n;
    signal[i] = static_cast<float>(std::sin(2 * M_PI * 5 * t) +
                                   0.5 * std::sin(2 * M_PI * 12 * t));
  }

  Array x = to_array(signal, Shape({n}), DType::F32, CPUPlace());
  printf("Input signal (first 8): ");
  for (int i = 0; i < 8; ++i)
    printf("%.3f ", x.at(i).item<float>());
  printf("...\n");

  // RFFT -> IRFFT roundtrip
  Array X = fft::rfft(x);
  printf("RFFT output length: %ld (complex)\n", X.numel());

  Array x_recon = fft::irfft(X, n);
  printf("Reconstructed signal (first 8): ");
  for (int i = 0; i < 8; ++i)
    printf("%.3f ", x_recon.at(i).item<float>());
  printf("...\n");

  // Check reconstruction error
  float max_err = 0;
  for (int i = 0; i < n; ++i) {
    float err = std::abs(x.at(i).item<float>() - x_recon.at(i).item<float>());
    if (err > max_err)
      max_err = err;
  }
  printf("Max reconstruction error: %e\n", max_err);

  // F64 FFT
  separator("CPU FFT (F64)");
  std::vector<double> signal_f64(n);
  for (int i = 0; i < n; ++i) {
    signal_f64[i] = std::cos(2 * M_PI * 3 * static_cast<double>(i) / n);
  }
  Array x64 = to_array(signal_f64, Shape({n}), DType::F64, CPUPlace());
  Array X64 = fft::rfft(x64);
  Array x64_recon = fft::irfft(X64, n);
  double max_err_f64 = 0;
  for (int i = 0; i < n; ++i) {
    double err =
        std::abs(x64.at(i).item<double>() - x64_recon.at(i).item<double>());
    if (err > max_err_f64)
      max_err_f64 = err;
  }
  printf("F64 FFT roundtrip max error: %e\n", max_err_f64);

  // next_fast_len
  separator("FFT Length Optimization");
  printf("next_fast_len(64)  = %ld\n", fft::next_fast_len(64));
  printf("next_fast_len(100) = %ld\n", fft::next_fast_len(100));
  printf("next_fast_len(127) = %ld\n", fft::next_fast_len(127));
}

static void run_fft_gpu() {
  separator("GPU FFT");

  int n = 64;
  std::vector<float> signal(n);
  for (int i = 0; i < n; ++i) {
    double t = static_cast<double>(i) / n;
    signal[i] = static_cast<float>(std::sin(2 * M_PI * 5 * t) +
                                   0.5 * std::sin(2 * M_PI * 12 * t));
  }

  Array x_cpu = to_array(signal, Shape({n}), DType::F32, CPUPlace());
  Array x = x_cpu.to(GPUPlace(0));

  Array X = fft::rfft(x);
  Array x_recon = fft::irfft(X, n);
  Array cpu_recon = x_recon.to(CPUPlace());

  printf("GPU RFFT->IRFFT roundtrip (first 8): ");
  for (int i = 0; i < 8; ++i)
    printf("%.3f ", cpu_recon.at(i).item<float>());
  printf("...\n");

  float max_err = 0;
  for (int i = 0; i < n; ++i) {
    float err = std::abs(signal[i] - cpu_recon.at(i).item<float>());
    if (err > max_err)
      max_err = err;
  }
  printf("GPU max reconstruction error: %e\n", max_err);

  // GPU F64 FFT
  separator("GPU FFT (F64)");
  std::vector<double> signal_f64(n);
  for (int i = 0; i < n; ++i)
    signal_f64[i] = std::cos(2 * M_PI * 3 * static_cast<double>(i) / n);
  Array x64_cpu = to_array(signal_f64, Shape({n}), DType::F64, CPUPlace());
  Array x64 = x64_cpu.to(GPUPlace(0));
  Array X64 = fft::rfft(x64);
  Array x64_recon = fft::irfft(X64, n).to(CPUPlace());
  double max_err_f64 = 0;
  for (int i = 0; i < n; ++i) {
    double err = std::abs(signal_f64[i] - x64_recon.at(i).item<double>());
    if (err > max_err_f64)
      max_err_f64 = err;
  }
  printf("GPU F64 FFT roundtrip max error: %e\n", max_err_f64);
}

int main() {
  ins::init();

  printf("Insight7 FFT Demo\n");
  printf("FFTW3: %s\n", is_compiled_with_fftw3() ? "yes" : "no");

  run_fft_cpu();

  if (gpu_available()) {
    run_fft_gpu();
  } else {
    printf("\n[GPU not available, skipping GPU FFT demo]\n");
  }

  printf("\nDone!\n");
  return 0;
}
