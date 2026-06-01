// src/ops/signal/radar.cpp
#include "insight/ops/signal/radar.h"
#include "insight/core/exception.h"
#include "insight/core/op_registry.h"
#include "insight/ops/cast.h"
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
    Array pulse = slice(x_cpu, {Slice(p, p + 1), Slice(0, samples_per_pulse)});
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
    Array col = slice(x_t_cpu, 0, s, s + 1);
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
  Array data_cpu = data.contiguous().to(cpu);

  int ndim = data.shape().ndim();
  INS_CHECK(ndim <= 2, "ca_cfar: supports 1D and 2D arrays only");

  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  data_cpu = data_cpu.to(work_dtype);

  double alpha = cfar_alpha(
      pfa, (int)reference_cells.size() > 0 ? (2 * reference_cells[0]) : 2);

  if (ndim == 2) {
    int N_row = 2 * reference_cells[0];
    int N_col = reference_cells.size() > 1 ? 2 * reference_cells[1] : N_row;
    alpha = cfar_alpha(pfa, N_row * N_col);
  }

  // Wrap scalar parameters as 1-element arrays for kernel dispatch
  Array alpha_arr =
      to_array(std::vector<double>{alpha}, Shape({1}), DType::F64, cpu);
  int g = guard_cells.empty() ? 1 : guard_cells[0];
  int r = reference_cells.empty() ? 1 : reference_cells[0];

  // guard_cells and reference_cells as int arrays
  // Kernel reads as int (not int64)
  Array gc_arr = zeros({1}, DType::I32, cpu);
  Array rc_arr = zeros({1}, DType::I32, cpu);
  // Write scalar values via layout pointer
  *(int *)gc_arr.layout_ptr()->data = g;
  *(int *)rc_arr.layout_ptr()->data = r;

  // Pre-allocate outputs
  Array threshold = zeros(data_cpu.shape(), work_dtype, cpu);
  Array detections = zeros(data_cpu.shape(), DType::BOOL, cpu);

  // Dispatch to backend kernel
  ops().launch("ca_cfar", cpu, work_dtype,
               {(void *)data_cpu.layout_ptr(), (void *)alpha_arr.layout_ptr(),
                (void *)gc_arr.layout_ptr(), (void *)rc_arr.layout_ptr()},
               {threshold.layout_ptr(), detections.layout_ptr()});

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
// ambgfun — Ambiguity Function (dispatches to backend kernel)
// ============================================================================

Array ambgfun(const Array &x, double fs, double prf, const Array &y,
              const std::string &cut, double cutValue) {
  INS_CHECK(x.defined(), "ambgfun: x is undefined");
  INS_CHECK(cut == "2d", "ambgfun: only '2d' cut is currently supported");

  Place cpu = CPUPlace();
  Array x_cpu = x.contiguous().to(cpu);
  Array y_cpu = y.defined() ? y.contiguous().to(cpu) : x_cpu;

  int64_t N = x_cpu.numel();
  int64_t M = y_cpu.numel();

  // Convert to complex128
  Array x_cpx = x_cpu;
  if (x_cpu.dtype() != DType::C64)
    x_cpx = to_complex(x_cpu);
  Array y_cpx = y_cpu;
  if (y_cpu.dtype() != DType::C64)
    y_cpx = to_complex(y_cpu);

  // Normalize using composite ops: x /= sqrt(sum(|x|^2))
  Array x_energy = sum(square(abs(x_cpx)));
  x_cpx = div(x_cpx, sqrt(x_energy));
  Array y_energy = sum(square(abs(y_cpx)));
  y_cpx = div(y_cpx, sqrt(y_energy));

  // Flatten to 1D
  x_cpx = reshape(x_cpx, {N});
  y_cpx = reshape(y_cpx, {M});

  // Build params array: [fs, N, M]
  Array params = to_array(std::vector<double>{fs, (double)N, (double)M},
                          Shape({3}), DType::F64, cpu);

  int64_t delay_len = N + M - 1;
  int64_t doppler_len = N;

  // Pre-allocate output
  Array result({doppler_len, delay_len}, DType::F64, cpu);

  // Dispatch to backend kernel
  ops().launch("ambgfun", cpu, DType::F64,
               {(void *)x_cpx.layout_ptr(), (void *)y_cpx.layout_ptr(),
                (void *)params.layout_ptr()},
               {result.layout_ptr()});

  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

} // namespace signal
} // namespace ins
