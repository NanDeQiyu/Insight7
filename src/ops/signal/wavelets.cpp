// src/ops/signal/wavelets.cpp
#include "insight/ops/signal/wavelets.h"
#include "insight/core/exception.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/signal/convolution.h"
#include "insight/ops/unary.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// ============================================================================
// morlet
// ============================================================================

Array morlet(int64_t M, double w, double s, bool complete) {
  INS_CHECK(M >= 1, "morlet: M must be >= 1");

  Place cpu = CPUPlace();
  std::vector<std::complex<double>> result(M);

  // Domain: [-s*2*pi, +s*2*pi]
  double lo = -s * 2.0 * M_PI;
  double hi = s * 2.0 * M_PI;
  double dt = (hi - lo) / (M - 1);

  for (int64_t i = 0; i < M; ++i) {
    double x = lo + i * dt;
    std::complex<double> val =
        std::exp(std::complex<double>(0, w * x)) * std::exp(-0.5 * x * x);
    if (complete) {
      val -= std::exp(-0.5 * w * w) * std::exp(-0.5 * x * x);
    }
    val *= std::pow(M_PI, -0.25);
    result[i] = val;
  }

  return to_array(result, DType::C64, cpu);
}

// ============================================================================
// ricker
// ============================================================================

Array ricker(int64_t points, double a) {
  INS_CHECK(points >= 1, "ricker: points must be >= 1");
  INS_CHECK(a > 0, "ricker: a must be > 0");

  Place cpu = CPUPlace();
  std::vector<double> result(points);

  double A = 2.0 / (std::sqrt(3.0 * a) * std::pow(M_PI, 0.25));

  for (int64_t i = 0; i < points; ++i) {
    double x = i - (points - 1.0) / 2.0;
    double xsq = x * x / (a * a);
    result[i] = A * (1.0 - xsq) * std::exp(-0.5 * xsq);
  }

  return to_array(result, DType::F64, cpu);
}

// ============================================================================
// morlet2
// ============================================================================

Array morlet2(int64_t M, double s, double w) {
  INS_CHECK(M >= 1, "morlet2: M must be >= 1");
  INS_CHECK(s > 0, "morlet2: s must be > 0");

  Place cpu = CPUPlace();
  std::vector<std::complex<double>> result(M);

  double coeff = std::pow(M_PI, -0.25) / std::sqrt(s);

  for (int64_t i = 0; i < M; ++i) {
    double x = (i - (M - 1.0) / 2.0) / s;
    std::complex<double> val = coeff *
                               std::exp(std::complex<double>(0, w * x)) *
                               std::exp(-0.5 * x * x);
    result[i] = val;
  }

  return to_array(result, DType::C64, cpu);
}

// ============================================================================
// cwt
// ============================================================================

Array cwt(const Array &data, std::function<Array(int64_t, double)> wavelet,
          const std::vector<double> &widths) {
  INS_CHECK(data.defined(), "cwt: data is undefined");
  INS_CHECK(!widths.empty(), "cwt: widths must not be empty");

  Place cpu = CPUPlace();
  Place dev = data.place();
  Array data_cpu = (dev.kind() == DeviceKind::CPU) ? data : data.to(cpu);

  int64_t n = data_cpu.numel();
  int64_t nw = static_cast<int64_t>(widths.size());

  // Determine output dtype (complex if wavelet returns complex)
  Array test_w = wavelet(10, widths[0]);
  bool is_complex =
      (test_w.dtype() == DType::C64 || test_w.dtype() == DType::C32);

  // Prepare data as complex for FFT convolution
  Array data_cpx = data_cpu;
  if (data_cpu.dtype() != DType::C64 && data_cpu.dtype() != DType::C32) {
    data_cpx = to_complex(data_cpu);
  }

  // Precompute FFT of data (once)
  Array data_fft = fft::fft(data_cpx, n);
  // Reshape to [1, n] for broadcasting with [nw, n]
  Array data_fft_2d = reshape(data_fft, {1, n});

  // --- Batched wavelet preparation ---
  // Pre-compute all wavelets, zero-pad to n, stack into [nw, n]
  std::vector<Array> w_rows;
  std::vector<int64_t> Ns; // original wavelet lengths for circular shift
  w_rows.reserve(nw);
  Ns.reserve(nw);

  for (int64_t ind = 0; ind < nw; ++ind) {
    double width = widths[ind];
    int64_t N = std::min(static_cast<int64_t>(10 * width), n);
    if (N < 1)
      N = 1;
    Ns.push_back(N);

    Array w_arr = wavelet(N, width);
    Array w_conj = conj(w_arr);
    Array w_rev = flip(w_conj, 0);

    if (w_rev.dtype() != DType::C64 && w_rev.dtype() != DType::C32) {
      w_rev = to_complex(w_rev);
    }

    // Zero-pad to length n
    int64_t pad_len = n - w_rev.numel();
    if (pad_len > 0) {
      w_rev = concat({w_rev, zeros({pad_len}, DType::C64, cpu)}, 0);
    } else if (pad_len < 0) {
      w_rev = slice(w_rev, 0, 0, n);
    }
    w_rows.push_back(w_rev);
  }

  // Stack into [nw, n] and do batched FFT along axis 1
  Array w_stack = stack(w_rows, 0);      // [nw, n]
  Array w_fft = fft::fft(w_stack, n, 1); // [nw, n] batched FFT

  // Batched multiply and inverse FFT
  Array conv_fft = mul(w_fft, data_fft_2d);      // [nw, n] broadcast
  Array conv_result = fft::ifft(conv_fft, n, 1); // [nw, n] batched IFFT

  // --- Extract "same" mode with per-row circular shift ---
  std::vector<Array> result_rows;
  result_rows.reserve(nw);

  for (int64_t ind = 0; ind < nw; ++ind) {
    int64_t start = Ns[ind] / 2;
    Array row = slice(conv_result, 0, ind, ind + 1); // [1, n]
    row = reshape(row, {n});                         // [n]

    // Circular shift by -start
    Array part1 = slice(row, 0, start, n);
    Array part2 = slice(row, 0, 0, start);
    Array conv_shifted = concat({part1, part2}, 0);

    if (conv_shifted.dtype() != DType::C64) {
      conv_shifted = as_complex(conv_shifted);
    }
    result_rows.push_back(conv_shifted);
  }

  Array result = stack(result_rows, 0);

  if (is_complex) {
    return result;
  }
  return real(result);
}

} // namespace signal
} // namespace ins
