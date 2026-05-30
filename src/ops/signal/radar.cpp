// src/ops/signal/radar.cpp
#include "insight/ops/signal/radar.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
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

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array tpl = (template_tx.place().kind() == DeviceKind::CPU)
                  ? template_tx
                  : template_tx.to(cpu);

  // Apply window to template if specified
  if (!window.empty()) {
    int64_t Nx = tpl.numel();
    Array W = get_window(window, Nx, false);
    tpl = mul(tpl, W);
  }

  // Normalize template if requested
  if (normalize) {
    double norm_val = 0.0;
    if (tpl.dtype() == DType::F64) {
      const double *d = tpl.data<double>();
      for (int64_t i = 0; i < tpl.numel(); ++i)
        norm_val += d[i] * d[i];
      norm_val = std::sqrt(norm_val);
      std::vector<double> norm_data(tpl.numel());
      for (int64_t i = 0; i < tpl.numel(); ++i)
        norm_data[i] = d[i] / norm_val;
      tpl = to_array(norm_data, DType::F64, cpu);
    } else {
      const float *d = tpl.data<float>();
      for (int64_t i = 0; i < tpl.numel(); ++i)
        norm_val += d[i] * d[i];
      norm_val = std::sqrt(norm_val);
      std::vector<float> norm_data(tpl.numel());
      for (int64_t i = 0; i < tpl.numel(); ++i)
        norm_data[i] = d[i] / norm_val;
      tpl = to_array(norm_data, DType::F32, cpu);
    }
  }

  // FFT of template, conjugate, tile to match num_pulses
  Array fft_tpl = fft::rfft(tpl, nfft);

  // Conjugate and tile
  int64_t fft_len = nfft / 2 + 1;
  std::vector<std::complex<double>> conj_tpl(fft_len);
  if (fft_tpl.dtype() == DType::C64) {
    const std::complex<double> *d =
        reinterpret_cast<const std::complex<double> *>(fft_tpl.data<char>());
    for (int64_t i = 0; i < fft_len; ++i)
      conj_tpl[i] = std::conj(d[i]);
  } else {
    const std::complex<float> *d =
        reinterpret_cast<const std::complex<float> *>(fft_tpl.data<char>());
    for (int64_t i = 0; i < fft_len; ++i)
      conj_tpl[i] = std::conj(d[i]);
  }

  // Process each pulse
  std::vector<std::complex<double>> result_data(num_pulses * nfft);
  for (int64_t p = 0; p < num_pulses; ++p) {
    // Extract pulse p
    std::vector<double> pulse_data(samples_per_pulse);
    if (x_cpu.dtype() == DType::F64) {
      const double *xd = x_cpu.data<double>();
      for (int64_t i = 0; i < samples_per_pulse; ++i)
        pulse_data[i] = xd[p * samples_per_pulse + i];
    } else {
      const float *xd = x_cpu.data<float>();
      for (int64_t i = 0; i < samples_per_pulse; ++i)
        pulse_data[i] = xd[p * samples_per_pulse + i];
    }

    Array pulse_arr = to_array(pulse_data, DType::F64, cpu);
    Array fft_pulse = fft::rfft(pulse_arr, nfft);

    // Multiply in frequency domain
    const std::complex<double> *fp =
        reinterpret_cast<const std::complex<double> *>(fft_pulse.data<char>());

    std::vector<std::complex<double>> prod(fft_len);
    for (int64_t i = 0; i < fft_len; ++i) {
      prod[i] = fp[i] * conj_tpl[i];
    }

    Array prod_arr = to_array(prod, DType::C64, cpu);
    Array ifft_result = fft::irfft(prod_arr, nfft);

    const double *ird = ifft_result.data<double>();
    for (int64_t i = 0; i < nfft; ++i) {
      result_data[p * nfft + i] = std::complex<double>(ird[i], 0.0);
    }
  }

  Array result =
      to_array(result_data, Shape({num_pulses, nfft}), DType::C64, cpu);
  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
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

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  // Apply window along pulse dimension if specified
  if (!window.empty()) {
    Array W = get_window(window, num_pulses, false);
    // Multiply each column by window
    if (x_cpu.dtype() == DType::F64) {
      const double *wd = W.data<double>();
      const double *xd = x_cpu.data<double>();
      std::vector<double> windowed(x_cpu.numel());
      for (int64_t p = 0; p < num_pulses; ++p) {
        for (int64_t s = 0; s < samples_per_pulse; ++s) {
          windowed[p * samples_per_pulse + s] =
              xd[p * samples_per_pulse + s] * wd[p];
        }
      }
      x_cpu = to_array(windowed, x_cpu.shape(), DType::F64, cpu);
    } else {
      const float *wd = W.data<float>();
      const float *xd = x_cpu.data<float>();
      std::vector<float> windowed(x_cpu.numel());
      for (int64_t p = 0; p < num_pulses; ++p) {
        for (int64_t s = 0; s < samples_per_pulse; ++s) {
          windowed[p * samples_per_pulse + s] =
              xd[p * samples_per_pulse + s] * wd[p];
        }
      }
      x_cpu = to_array(windowed, x_cpu.shape(), DType::F32, cpu);
    }
  }

  // FFT along pulse dimension (axis 0)
  // We need to transpose, FFT along last axis, then transpose back
  // Or do FFT column-by-column
  std::vector<std::complex<double>> result_data(nfft * samples_per_pulse);

  for (int64_t s = 0; s < samples_per_pulse; ++s) {
    // Extract column s
    std::vector<double> col(num_pulses);
    if (x_cpu.dtype() == DType::F64) {
      const double *xd = x_cpu.data<double>();
      for (int64_t p = 0; p < num_pulses; ++p)
        col[p] = xd[p * samples_per_pulse + s];
    } else {
      const float *xd = x_cpu.data<float>();
      for (int64_t p = 0; p < num_pulses; ++p)
        col[p] = xd[p * samples_per_pulse + s];
    }

    Array col_arr = to_array(col, DType::F64, cpu);
    Array fft_col = fft::fft(col_arr, nfft);

    const std::complex<double> *fd =
        reinterpret_cast<const std::complex<double> *>(fft_col.data<char>());
    for (int64_t p = 0; p < nfft; ++p) {
      result_data[p * samples_per_pulse + s] = fd[p];
    }
  }

  Array result =
      to_array(result_data, Shape({nfft, samples_per_pulse}), DType::C64, cpu);
  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
  return result;
}

// ============================================================================
// cfar_alpha
// ============================================================================

double cfar_alpha(double pfa, int N) {
  return N * (std::pow(pfa, -1.0 / N) - 1.0);
}

// ============================================================================
// ca_cfar
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

  double alpha = cfar_alpha(
      pfa, (int)reference_cells.size() > 0 ? (2 * reference_cells[0]) : 2);

  // For 2D: alpha is product of per-dimension alphas
  if (ndim == 2) {
    int N_row = 2 * reference_cells[0];
    int N_col = reference_cells.size() > 1 ? 2 * reference_cells[1] : N_row;
    alpha = cfar_alpha(pfa, N_row * N_col);
  }

  int64_t total = data_cpu.numel();
  Array threshold = zeros(data.shape(), data.dtype(), cpu);
  Array detections = zeros(data.shape(), DType::BOOL, cpu);

  if (ndim == 1) {
    int64_t n = data.shape().dim(0);
    int g = guard_cells.empty() ? 1 : guard_cells[0];
    int r = reference_cells.empty() ? 1 : reference_cells[0];

    if (data.dtype() == DType::F64) {
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
    } else if (data.dtype() == DType::F32) {
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

    // 2D CFAR using 2D cumsum
    if (data.dtype() == DType::F64) {
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
          // Reference window: cells between guard and reference boundaries
          int64_t r0 = std::max((int64_t)0, r - gr - rr);
          int64_t r1 = std::max((int64_t)0, r - gr);
          int64_t r2 = std::min(rows, r + gr + 1);
          int64_t r3 = std::min(rows, r + gr + rr + 1);
          int64_t c0 = std::max((int64_t)0, c - gc - rc);
          int64_t c1 = std::max((int64_t)0, c - gc);
          int64_t c2 = std::min(cols, c + gc + 1);
          int64_t c3 = std::min(cols, c + gc + rc + 1);

          // Sum of reference cells (outer region minus guard region)
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
    } else if (data.dtype() == DType::F32) {
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
// mvdr
// ============================================================================

Array mvdr(const Array &x, const Array &sv, bool calc_cov) {
  INS_CHECK(x.defined(), "mvdr: x is undefined");
  INS_CHECK(sv.defined(), "mvdr: sv is undefined");

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array sv_cpu = (sv.place().kind() == DeviceKind::CPU) ? sv : sv.to(cpu);

  INS_CHECK(x_cpu.shape().dim(0) <= x_cpu.shape().dim(1),
            "mvdr: matrix has more sensors than samples");
  INS_CHECK(x_cpu.shape().dim(0) == sv_cpu.numel(),
            "mvdr: steering vector and input data do not align");

  // Compute covariance matrix if needed
  Array R;
  if (calc_cov) {
    // R = (1/N) * X * X^H
    int64_t M = x_cpu.shape().dim(0);
    int64_t N = x_cpu.shape().dim(1);

    // Convert to complex for covariance computation
    // For simplicity, compute as real: R = X * X^T / N
    std::vector<double> R_data(M * M, 0.0);
    if (x_cpu.dtype() == DType::F64) {
      const double *xd = x_cpu.data<double>();
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < M; ++j) {
          double sum = 0.0;
          for (int64_t k = 0; k < N; ++k) {
            sum += xd[i * N + k] * xd[j * N + k];
          }
          R_data[i * M + j] = sum / N;
        }
      }
    } else {
      const float *xd = x_cpu.data<float>();
      for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < M; ++j) {
          double sum = 0.0;
          for (int64_t k = 0; k < N; ++k) {
            sum += static_cast<double>(xd[i * N + k]) * xd[j * N + k];
          }
          R_data[i * M + j] = sum / N;
        }
      }
    }
    R = to_array(R_data, Shape({M, M}), DType::F64, cpu);
  } else {
    R = x_cpu;
  }

  // R_inv = inv(R)
  Array R_inv = inv(R);

  // sv as column vector
  Array sv_col;
  if (sv_cpu.shape().ndim() == 1) {
    // Reshape to [M, 1]
    std::vector<double> sv_data(sv_cpu.numel());
    if (sv_cpu.dtype() == DType::F64) {
      const double *sd = sv_cpu.data<double>();
      for (int64_t i = 0; i < sv_cpu.numel(); ++i)
        sv_data[i] = sd[i];
    } else {
      const float *sd = sv_cpu.data<float>();
      for (int64_t i = 0; i < sv_cpu.numel(); ++i)
        sv_data[i] = sd[i];
    }
    sv_col = to_array(sv_data, Shape({sv_cpu.numel(), 1}), DType::F64, cpu);
  } else {
    sv_col = sv_cpu;
  }

  // w = R_inv * sv / (sv^H * R_inv * sv)
  Array wB = matmul(R_inv, sv_col);

  // sv^T (transpose for real case)
  std::vector<double> sv_t_data(sv_col.numel());
  const double *svd = sv_col.data<double>();
  for (int64_t i = 0; i < sv_col.numel(); ++i)
    sv_t_data[i] = svd[i];
  Array sv_t = to_array(sv_t_data, Shape({1, sv_col.numel()}), DType::F64, cpu);

  Array wA = matmul(sv_t, wB);

  // w = wB / wA (scalar division)
  double wA_val = 0.0;
  if (wA.dtype() == DType::F64) {
    wA_val = wA.data<double>()[0];
  } else {
    wA_val = wA.data<float>()[0];
  }

  std::vector<double> w_data(wB.numel());
  const double *wb = wB.data<double>();
  for (int64_t i = 0; i < wB.numel(); ++i) {
    w_data[i] = wb[i] / wA_val;
  }
  Array result = to_array(w_data, DType::F64, cpu);

  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
  return result;
}

// ============================================================================
// ambgfun
// ============================================================================

Array ambgfun(const Array &x, double fs, double prf, const Array &y,
              const std::string &cut, double cutValue) {
  INS_CHECK(x.defined(), "ambgfun: x is undefined");

  Place cpu = CPUPlace();
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  Array y_cpu;
  if (y.defined()) {
    y_cpu = (y.place().kind() == DeviceKind::CPU) ? y : y.to(cpu);
  } else {
    y_cpu = x_cpu;
  }

  int64_t N = x_cpu.numel();
  int64_t M = y_cpu.numel();

  // Compute ambiguity function
  // ambgfun is essentially: |sum_n x(n) * conj(y(n-tau)) * exp(j*2*pi*f*n/fs)|
  // For the 2D case, we compute for all delay and Doppler values

  if (cut == "2d") {
    int64_t delay_len = N + M - 1;
    int64_t doppler_len = N;

    std::vector<double> ambg(delay_len * doppler_len, 0.0);

    // Get data as complex
    std::vector<std::complex<double>> x_data(N), y_data(M);
    if (x_cpu.dtype() == DType::C64) {
      const std::complex<double> *xd =
          reinterpret_cast<const std::complex<double> *>(x_cpu.data<char>());
      for (int64_t i = 0; i < N; ++i)
        x_data[i] = xd[i];
    } else if (x_cpu.dtype() == DType::F64) {
      const double *xd = x_cpu.data<double>();
      for (int64_t i = 0; i < N; ++i)
        x_data[i] = std::complex<double>(xd[i], 0.0);
    } else if (x_cpu.dtype() == DType::C32) {
      const std::complex<float> *xd =
          reinterpret_cast<const std::complex<float> *>(x_cpu.data<char>());
      for (int64_t i = 0; i < N; ++i)
        x_data[i] = xd[i];
    } else {
      const float *xd = x_cpu.data<float>();
      for (int64_t i = 0; i < N; ++i)
        x_data[i] = std::complex<double>(xd[i], 0.0);
    }

    if (y_cpu.dtype() == DType::C64) {
      const std::complex<double> *yd =
          reinterpret_cast<const std::complex<double> *>(y_cpu.data<char>());
      for (int64_t i = 0; i < M; ++i)
        y_data[i] = yd[i];
    } else if (y_cpu.dtype() == DType::F64) {
      const double *yd = y_cpu.data<double>();
      for (int64_t i = 0; i < M; ++i)
        y_data[i] = std::complex<double>(yd[i], 0.0);
    } else if (y_cpu.dtype() == DType::C32) {
      const std::complex<float> *yd =
          reinterpret_cast<const std::complex<float> *>(y_cpu.data<char>());
      for (int64_t i = 0; i < M; ++i)
        y_data[i] = yd[i];
    } else {
      const float *yd = y_cpu.data<float>();
      for (int64_t i = 0; i < M; ++i)
        y_data[i] = std::complex<double>(yd[i], 0.0);
    }

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

    // For each Doppler frequency
    for (int64_t fd = 0; fd < doppler_len; ++fd) {
      double freq = (fd - doppler_len / 2.0) * fs / doppler_len;
      double omega = 2.0 * M_PI * freq / fs;

      // For each delay
      for (int64_t tau = 0; tau < delay_len; ++tau) {
        int64_t tau_shift = tau - (M - 1);

        std::complex<double> sum(0.0, 0.0);
        for (int64_t n = 0; n < N; ++n) {
          int64_t m = n - tau_shift;
          if (m >= 0 && m < M) {
            double phase = omega * n;
            sum += x_data[n] * std::conj(y_data[m]) *
                   std::exp(std::complex<double>(0.0, phase));
          }
        }
        ambg[fd * delay_len + tau] = std::abs(sum);
      }
    }

    // Normalize peak to 1
    double peak = 0;
    for (auto &v : ambg)
      peak = std::max(peak, v);
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

  // For "delay" and "doppler" cuts, compute 1D slices
  INS_THROW("ambgfun: only '2d' cut is currently supported");
}

} // namespace signal
} // namespace ins
