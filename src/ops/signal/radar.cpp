// src/ops/signal/radar.cpp
#include "insight/ops/signal/radar.h"
#include "insight/core/exception.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal/windows.h"
#include "insight/ops/unary.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

// ============================================================================
// pulse_compression
// ============================================================================

Array pulse_compression(const Array &x, const Array &template_tx,
                        bool normalize, const std::string &window,
                        int64_t nfft) {
  INS_CHECK(x.defined(), "pulse_compression: x is undefined");
  INS_CHECK(template_tx.defined(),
            "pulse_compression: template_tx is undefined");
  INS_CHECK(x.shape().ndim() == 2,
            "pulse_compression: x must be 2D [num_pulses, samples_per_pulse]");
  INS_CHECK(template_tx.shape().ndim() == 1,
            "pulse_compression: template_tx must be 1D");

  int64_t num_pulses = x.shape().dim(0);
  int64_t samples_per_pulse = x.shape().dim(1);

  if (nfft == 0)
    nfft = samples_per_pulse;

  // Apply window to template if specified
  Array tpl = template_tx;
  if (!window.empty()) {
    Array W = get_window(window, tpl.numel(), false);
    tpl = mul(tpl, W);
  }

  // Normalize template
  if (normalize) {
    double norm_val = std::sqrt(sum(square(tpl)).item<double>());
    tpl = div(tpl, full(tpl.shape(), norm_val, tpl.dtype(), tpl.place()));
  }

  // FFT of template, conjugate
  DType work_dtype = (tpl.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (tpl.dtype() != work_dtype)
    tpl = tpl.to(work_dtype);
  Array fft_tpl = fft::rfft(tpl, nfft);
  Array conj_fft_tpl = conj(fft_tpl);

  // Process each pulse: rfft → multiply → irfft
  Place cpu = CPUPlace();
  int64_t fft_len = nfft / 2 + 1;
  DType cdtype = (work_dtype == DType::F32) ? DType::C32 : DType::C64;

  // Move all data to CPU for per-pulse extraction
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  if (x_cpu.dtype() != work_dtype)
    x_cpu = x_cpu.to(work_dtype);
  if (conj_fft_tpl.place().kind() != DeviceKind::CPU)
    conj_fft_tpl = conj_fft_tpl.to(cpu);

  std::vector<Array> result_rows;

  for (int64_t p = 0; p < num_pulses; ++p) {
    // Extract pulse p using slice
    Array pulse =
        slice(x_cpu, {0, 1}, {static_cast<int>(p), 0},
              {static_cast<int>(p + 1), static_cast<int>(samples_per_pulse)});
    pulse = reshape(pulse, {samples_per_pulse});
    if (pulse.dtype() != work_dtype)
      pulse = pulse.to(work_dtype);

    Array fft_pulse = fft::rfft(pulse, nfft);
    Array product = mul(fft_pulse, conj_fft_tpl);
    Array ifft_result = fft::irfft(product, nfft);

    // Convert to complex for storage
    Array ifft_cpx = to_complex(ifft_result);
    result_rows.push_back(ifft_cpx);
  }

  Array result = stack(result_rows, 0);
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// pulse_doppler
// ============================================================================

Array pulse_doppler(const Array &x, const std::string &window, int64_t nfft) {
  INS_CHECK(x.defined(), "pulse_doppler: x is undefined");
  INS_CHECK(x.shape().ndim() == 2,
            "pulse_doppler: x must be 2D [num_pulses, samples_per_pulse]");

  int64_t num_pulses = x.shape().dim(0);
  int64_t samples_per_pulse = x.shape().dim(1);

  if (nfft == 0)
    nfft = num_pulses;

  // Apply window along pulse dimension (axis 0)
  Array x_work = x;
  if (!window.empty()) {
    Array W = get_window(window, num_pulses, false);
    // Reshape W for broadcasting: [num_pulses, 1]
    W = reshape(W, {num_pulses, 1});
    x_work = mul(x_work, W);
  }

  // FFT along pulse dimension (axis 0)
  // Transpose → FFT along last axis → transpose back
  Array x_t = transpose(x_work); // [samples_per_pulse, num_pulses]

  // Convert to complex for FFT
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (x_t.dtype() != work_dtype)
    x_t = x_t.to(work_dtype);

  // FFT each row (which corresponds to a column of original)
  // Process row by row
  Place cpu = CPUPlace();
  Array x_t_cpu = (x_t.place().kind() == DeviceKind::CPU) ? x_t : x_t.to(cpu);

  std::vector<Array> col_results;
  for (int64_t s = 0; s < samples_per_pulse; ++s) {
    Array col =
        slice(x_t_cpu, {0}, {static_cast<int>(s)}, {static_cast<int>(s + 1)});
    col = reshape(col, {num_pulses});

    Array col_cpx = to_complex(col);
    Array fft_col = fft::fft(col_cpx, nfft);
    col_results.push_back(fft_col);
  }

  // Stack columns and transpose back to [nfft, samples_per_pulse]
  Array stacked = stack(col_results, 0); // [samples_per_pulse, nfft]
  Array result = transpose(stacked);     // [nfft, samples_per_pulse]
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// cfar_alpha
// ============================================================================

double cfar_alpha(double pfa, int N) {
  return N * (std::pow(pfa, -1.0 / N) - 1.0);
}

// ============================================================================
// ca_cfar — Cell-Averaging CFAR (sequential, requires backend kernel)
// ============================================================================

std::pair<Array, Array> ca_cfar(const Array &data,
                                const std::vector<int> &guard_cells,
                                const std::vector<int> &reference_cells,
                                double pfa) {
  INS_CHECK(data.defined(), "ca_cfar: input is undefined");

  Place cpu = CPUPlace();
  Array data_cpu =
      (data.place().kind() == DeviceKind::CPU) ? data : data.to(cpu);

  int ndim = data.shape().ndim();
  INS_CHECK(ndim <= 2, "ca_cfar: supports 1D and 2D arrays only");

  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;

  double alpha = cfar_alpha(
      pfa, (int)reference_cells.size() > 0 ? (2 * reference_cells[0]) : 2);

  if (ndim == 2) {
    int N_row = 2 * reference_cells[0];
    int N_col = reference_cells.size() > 1 ? 2 * reference_cells[1] : N_row;
    alpha = cfar_alpha(pfa, N_row * N_col);
  }

  int64_t total = data_cpu.numel();
  Array threshold = zeros(data.shape(), work_dtype, cpu);
  Array detections = zeros(data.shape(), DType::BOOL, cpu);

  if (ndim == 1) {
    int64_t n = data.shape().dim(0);
    int g = guard_cells.empty() ? 1 : guard_cells[0];
    int r = reference_cells.empty() ? 1 : reference_cells[0];

    if (work_dtype == DType::F64) {
      const double *src = data_cpu.data<double>();
      double *th = threshold.data<double>();
      bool *det = detections.data<bool>();

      // Use cumsum for efficient cell averaging
      std::vector<double> cumsum(n + 1, 0.0);
      for (int64_t i = 0; i < n; ++i)
        cumsum[i + 1] = cumsum[i] + src[i];

      for (int64_t i = 0; i < n; ++i) {
        int64_t left_start = std::max((int64_t)0, i - g - r);
        int64_t left_end = std::max((int64_t)0, i - g);
        int64_t right_start = std::min(n, i + g + 1);
        int64_t right_end = std::min(n, i + g + r + 1);

        int64_t count = (left_end - left_start) + (right_end - right_start);
        if (count == 0) {
          th[i] = 0;
          det[i] = false;
          continue;
        }

        double sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                        (cumsum[right_end] - cumsum[right_start]);
        double noise_level = sum_lr / count;
        th[i] = noise_level * alpha;
        det[i] = src[i] > th[i];
      }
    } else {
      const float *src = data_cpu.data<float>();
      float *th = threshold.data<float>();
      bool *det = detections.data<bool>();

      std::vector<float> cumsum(n + 1, 0.0f);
      for (int64_t i = 0; i < n; ++i)
        cumsum[i + 1] = cumsum[i] + src[i];

      for (int64_t i = 0; i < n; ++i) {
        int64_t left_start = std::max((int64_t)0, i - g - r);
        int64_t left_end = std::max((int64_t)0, i - g);
        int64_t right_start = std::min(n, i + g + 1);
        int64_t right_end = std::min(n, i + g + r + 1);

        int64_t count = (left_end - left_start) + (right_end - right_start);
        if (count == 0) {
          th[i] = 0;
          det[i] = false;
          continue;
        }

        float sum_lr = (cumsum[left_end] - cumsum[left_start]) +
                       (cumsum[right_end] - cumsum[right_start]);
        float noise_level = sum_lr / count;
        th[i] = noise_level * static_cast<float>(alpha);
        det[i] = src[i] > th[i];
      }
    }
  } else if (ndim == 2) {
    int64_t rows = data.shape().dim(0);
    int64_t cols = data.shape().dim(1);
    int gr = guard_cells.empty() ? 1 : guard_cells[0];
    int gc = guard_cells.size() > 1 ? guard_cells[1] : gr;
    int rr = reference_cells.empty() ? 1 : reference_cells[0];
    int rc = reference_cells.size() > 1 ? reference_cells[1] : rr;

    if (work_dtype == DType::F64) {
      const double *src = data_cpu.data<double>();
      double *th = threshold.data<double>();
      bool *det = detections.data<bool>();

      // Build 2D cumsum
      std::vector<double> cs((rows + 1) * (cols + 1), 0.0);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          cs[(r + 1) * (cols + 1) + (c + 1)] =
              src[r * cols + c] + cs[r * (cols + 1) + (c + 1)] +
              cs[(r + 1) * (cols + 1) + c] - cs[r * (cols + 1) + c];
        }
      }

      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          int64_t r0 = std::max((int64_t)0, r - gr - rr);
          int64_t r1 = std::max((int64_t)0, r - gr);
          int64_t r2 = std::min(rows, r + gr + 1);
          int64_t r3 = std::min(rows, r + gr + rr + 1);
          int64_t c0 = std::max((int64_t)0, c - gc - rc);
          int64_t c1 = std::max((int64_t)0, c - gc);
          int64_t c2 = std::min(cols, c + gc + 1);
          int64_t c3 = std::min(cols, c + gc + rc + 1);

          double outer = cs[r3 * (cols + 1) + c3] - cs[r0 * (cols + 1) + c3] -
                         cs[r3 * (cols + 1) + c0] + cs[r0 * (cols + 1) + c0];
          double inner = cs[r2 * (cols + 1) + c2] - cs[r1 * (cols + 1) + c2] -
                         cs[r2 * (cols + 1) + c1] + cs[r1 * (cols + 1) + c1];
          double ref_sum = outer - inner;

          int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));
          if (ref_count <= 0) {
            th[r * cols + c] = 0;
            det[r * cols + c] = false;
            continue;
          }

          double noise_level = ref_sum / ref_count;
          th[r * cols + c] = noise_level * alpha;
          det[r * cols + c] = src[r * cols + c] > th[r * cols + c];
        }
      }
    } else {
      const float *src = data_cpu.data<float>();
      float *th = threshold.data<float>();
      bool *det = detections.data<bool>();

      std::vector<float> cs((rows + 1) * (cols + 1), 0.0f);
      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          cs[(r + 1) * (cols + 1) + (c + 1)] =
              src[r * cols + c] + cs[r * (cols + 1) + (c + 1)] +
              cs[(r + 1) * (cols + 1) + c] - cs[r * (cols + 1) + c];
        }
      }

      for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
          int64_t r0 = std::max((int64_t)0, r - gr - rr);
          int64_t r1 = std::max((int64_t)0, r - gr);
          int64_t r2 = std::min(rows, r + gr + 1);
          int64_t r3 = std::min(rows, r + gr + rr + 1);
          int64_t c0 = std::max((int64_t)0, c - gc - rc);
          int64_t c1 = std::max((int64_t)0, c - gc);
          int64_t c2 = std::min(cols, c + gc + 1);
          int64_t c3 = std::min(cols, c + gc + rc + 1);

          float outer = cs[r3 * (cols + 1) + c3] - cs[r0 * (cols + 1) + c3] -
                        cs[r3 * (cols + 1) + c0] + cs[r0 * (cols + 1) + c0];
          float inner = cs[r2 * (cols + 1) + c2] - cs[r1 * (cols + 1) + c2] -
                        cs[r2 * (cols + 1) + c1] + cs[r1 * (cols + 1) + c1];
          float ref_sum = outer - inner;

          int64_t ref_count = ((r3 - r0) * (c3 - c0)) - ((r2 - r1) * (c2 - c1));
          if (ref_count <= 0) {
            th[r * cols + c] = 0;
            det[r * cols + c] = false;
            continue;
          }

          float noise_level = ref_sum / ref_count;
          th[r * cols + c] = noise_level * static_cast<float>(alpha);
          det[r * cols + c] = src[r * cols + c] > th[r * cols + c];
        }
      }
    }
  }

  if (data.place().kind() != DeviceKind::CPU) {
    threshold = threshold.to(data.place());
    detections = detections.to(data.place());
  }
  return {threshold, detections};
}

// ============================================================================
// mvdr — MVDR Beamformer using composite ops
// ============================================================================

Array mvdr(const Array &x, const Array &sv, bool calc_cov) {
  INS_CHECK(x.defined(), "mvdr: x is undefined");
  INS_CHECK(sv.defined(), "mvdr: sv is undefined");

  INS_CHECK(x.shape().dim(0) <= x.shape().dim(1),
            "mvdr: matrix has more sensors than samples");
  INS_CHECK(x.shape().dim(0) == sv.numel(),
            "mvdr: steering vector and input data do not align");

  Place cpu = CPUPlace();
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;

  // Compute covariance matrix if needed: R = (1/N) * X * X^T
  Array R;
  if (calc_cov) {
    int64_t N = x.shape().dim(1);
    Array x_work = x.dtype() == work_dtype ? x : x.to(work_dtype);
    Array xT = transpose(x_work);
    R = div(matmul(x_work, xT), full({1}, static_cast<double>(N), work_dtype));
  } else {
    R = x.dtype() == work_dtype ? x : x.to(work_dtype);
  }

  // R_inv = inv(R)
  Array R_inv = inv(R);

  // sv as column vector
  Array sv_work = sv.dtype() == work_dtype ? sv : sv.to(work_dtype);
  Array sv_col;
  if (sv_work.shape().ndim() == 1) {
    sv_col = reshape(sv_work, {sv_work.numel(), 1});
  } else {
    sv_col = sv_work;
  }

  // w = R_inv * sv / (sv^T * R_inv * sv)
  Array wB = matmul(R_inv, sv_col);
  Array sv_t = transpose(sv_col);
  Array wA = matmul(sv_t, wB);

  // Scalar division using item()
  double wA_val = (work_dtype == DType::F64)
                      ? wA.item<double>()
                      : static_cast<double>(wA.item<float>());

  Array result = div(wB, full(wB.shape(), wA_val, work_dtype));
  result = reshape(result, {sv.numel()});

  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// ambgfun — Ambiguity Function (sequential, requires backend kernel)
// ============================================================================

Array ambgfun(const Array &x, double fs, double prf, const Array &y,
              const std::string &cut, double cutValue) {
  INS_CHECK(x.defined(), "ambgfun: x is undefined");

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array y_cpu = y.defined()
                    ? ((y.place().kind() == DeviceKind::CPU) ? y : y.to(cpu))
                    : x_cpu;

  int64_t N = x_cpu.numel();
  int64_t M = y_cpu.numel();

  INS_CHECK(cut == "2d", "ambgfun: only '2d' cut is currently supported");

  // Convert to complex vectors
  std::vector<std::complex<double>> x_data(N), y_data(M);
  auto to_cpx = [](const Array &arr, std::vector<std::complex<double>> &out) {
    int64_t n = arr.numel();
    if (arr.dtype() == DType::C64) {
      const std::complex<double> *d =
          reinterpret_cast<const std::complex<double> *>(arr.data<char>());
      for (int64_t i = 0; i < n; ++i)
        out[i] = d[i];
    } else if (arr.dtype() == DType::C32) {
      const std::complex<float> *d =
          reinterpret_cast<const std::complex<float> *>(arr.data<char>());
      for (int64_t i = 0; i < n; ++i)
        out[i] = d[i];
    } else if (arr.dtype() == DType::F64) {
      const double *d = arr.data<double>();
      for (int64_t i = 0; i < n; ++i)
        out[i] = {d[i], 0.0};
    } else {
      const float *d = arr.data<float>();
      for (int64_t i = 0; i < n; ++i)
        out[i] = {d[i], 0.0};
    }
  };
  to_cpx(x_cpu, x_data);
  to_cpx(y_cpu, y_data);

  // Normalize
  double x_norm = 0, y_norm = 0;
  for (auto &v : x_data)
    x_norm += std::norm(v);
  for (auto &v : y_data)
    y_norm += std::norm(v);
  x_norm = std::sqrt(x_norm);
  y_norm = std::sqrt(y_norm);
  for (auto &v : x_data)
    v /= x_norm;
  for (auto &v : y_data)
    v /= y_norm;

  int64_t delay_len = N + M - 1;
  int64_t doppler_len = N;
  std::vector<double> ambg(delay_len * doppler_len, 0.0);

  for (int64_t fd = 0; fd < doppler_len; ++fd) {
    double freq = (fd - doppler_len / 2.0) * fs / doppler_len;
    double omega = 2.0 * M_PI * freq / fs;

    for (int64_t tau = 0; tau < delay_len; ++tau) {
      int64_t tau_shift = tau - (M - 1);

      std::complex<double> sum_val(0.0, 0.0);
      for (int64_t n = 0; n < N; ++n) {
        int64_t m = n - tau_shift;
        if (m >= 0 && m < M) {
          double phase = omega * n;
          sum_val += x_data[n] * std::conj(y_data[m]) *
                     std::exp(std::complex<double>(0.0, phase));
        }
      }
      ambg[fd * delay_len + tau] = std::abs(sum_val);
    }
  }

  // Normalize peak to 1
  double peak = *std::max_element(ambg.begin(), ambg.end());
  if (peak > 0) {
    for (auto &v : ambg)
      v /= peak;
  }

  Array result =
      to_array(ambg, Shape({doppler_len, delay_len}), DType::F64, cpu);
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

} // namespace signal
} // namespace ins
