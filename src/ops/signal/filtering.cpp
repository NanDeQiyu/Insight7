// src/ops/signal/filtering.cpp
#include "insight/ops/signal/filtering.h"
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
#include "insight/ops/signal.h"
#include "insight/ops/signal/convolution.h"
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
// hilbert
// ============================================================================

Array hilbert(const Array &x, int64_t N) {
  INS_CHECK(x.defined(), "hilbert: input is undefined");
  INS_CHECK(x.dtype() == DType::F32 || x.dtype() == DType::F64,
            "hilbert: only float types supported");

  if (N < 0)
    N = x.numel();
  INS_CHECK(N >= 1, "hilbert: N must be >= 1");

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // FFT of input
  Array Xf = fft::fft(x, N);

  // Build frequency-domain Hilbert transform filter:
  // h[0] = 1, h[1..N/2-1] = 2, h[N/2] = 1 (if N even), h[N/2+1..N-1] = 0
  std::vector<std::complex<double>> h_data(N, {0.0, 0.0});
  h_data[0] = {1.0, 0.0};
  for (int64_t i = 1; i < (N + 1) / 2; ++i) {
    h_data[i] = {2.0, 0.0};
  }
  if (N % 2 == 0) {
    h_data[N / 2] = {1.0, 0.0};
  }

  Array h = to_array(h_data, cdtype, cpu);
  if (Xf.place().kind() != DeviceKind::CPU) {
    h = h.to(Xf.place());
  }

  // Multiply in frequency domain and inverse FFT
  Array result = fft::ifft(mul(Xf, h), N);
  return result;
}

// ============================================================================
// hilbert2
// ============================================================================

Array hilbert2(const Array &x, int64_t N) {
  INS_CHECK(x.defined(), "hilbert2: input is undefined");
  INS_CHECK(x.shape().ndim() == 2, "hilbert2: input must be 2D");

  int64_t rows = x.shape().dim(0);
  int64_t cols = x.shape().dim(1);

  if (N < 0)
    N = std::max(rows, cols);

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // Build 1D Hilbert filter
  auto make_h = [](int64_t n) {
    std::vector<std::complex<double>> h(n, {0.0, 0.0});
    h[0] = {1.0, 0.0};
    for (int64_t i = 1; i < (n + 1) / 2; ++i)
      h[i] = {2.0, 0.0};
    if (n % 2 == 0)
      h[n / 2] = {1.0, 0.0};
    return h;
  };

  // Build 2D filter as outer product
  auto h_row = make_h(N);
  auto h_col = make_h(N);
  std::vector<std::complex<double>> h2d(N * N);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      h2d[i * N + j] = h_row[i] * h_col[j];
    }
  }

  Array h_arr = to_array(h2d, Shape({N, N}), cdtype, cpu);
  Array Xf = fft::fft2(x, {N, N});

  if (Xf.place().kind() != DeviceKind::CPU) {
    h_arr = h_arr.to(Xf.place());
  }

  Array result = fft::ifft2(mul(Xf, h_arr));
  // Crop back to original size
  result = slice(result, {0}, {0}, {rows});
  result = slice(result, {1}, {0}, {cols});
  return result;
}

// ============================================================================
// detrend
// ============================================================================

Array detrend(const Array &data, int axis, const std::string &type) {
  INS_CHECK(data.defined(), "detrend: input is undefined");
  INS_CHECK(type == "linear" || type == "constant",
            "detrend: type must be 'linear' or 'constant'");

  int ndim = data.shape().ndim();
  int ax = axis;
  if (ax < 0)
    ax += ndim;
  INS_CHECK(ax >= 0 && ax < ndim, "detrend: axis out of range");

  if (type == "constant") {
    Array m = mean(data, ax, true);
    return sub(data, m);
  }

  // Linear detrend: subtract best-fit line along axis
  int64_t n = data.shape().dim(ax);
  if (n <= 1)
    return data.copy();

  double t_mean = (n - 1.0) / 2.0;
  double t_var = (n * n - 1.0) / 12.0;

  Array x_mean = mean(data, ax, true);

  // Build t array with correct shape for broadcasting
  Array t = arange(0.0, static_cast<double>(n), 1.0, DType::F64);
  std::vector<int64_t> t_shape(ndim, 1);
  t_shape[ax] = n;
  t = reshape(t, Shape(t_shape));

  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (t.dtype() != work_dtype)
    t = t.to(work_dtype);

  Array t_centered = sub(t, full(Shape(t_shape), t_mean, work_dtype));
  Array x_centered =
      sub(data.dtype() == work_dtype ? data : data.to(work_dtype),
          x_mean.dtype() == work_dtype ? x_mean : x_mean.to(work_dtype));

  Array product = mul(t_centered, x_centered);
  Array sum_product = sum(product, ax, true);

  double denom = n * t_var;
  Array slope = div(sum_product, full(sum_product.shape(), denom, work_dtype));
  Array trend = add(mul(slope, t_centered), x_mean);

  Array result =
      sub(data.dtype() == work_dtype ? data : data.to(work_dtype), trend);
  if (result.dtype() != data.dtype()) {
    result = result.to(data.dtype());
  }
  return result;
}

// ============================================================================
// firfilter — FIR filtering via convolution
// ============================================================================

Array firfilter(const Array &b, const Array &x, int axis) {
  INS_CHECK(b.defined() && x.defined(), "firfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1, "firfilter: b must be 1D");

  // For 1D input, use convolve directly
  if (x.shape().ndim() == 1) {
    return convolve(x, b, "full");
  }

  // For multi-dimensional: apply convolve along each slice of the axis
  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;

  // Move target axis to last position
  Array x_moved = x;
  if (ax != ndim - 1) {
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    x_moved = permute(x_moved, perm);
  }

  // Reshape to 2D: (batch, signal_len)
  int64_t batch_size = x_moved.numel() / x.shape().dim(ax);
  int64_t signal_len = x.shape().dim(ax);
  Array x_2d = reshape(x_moved, {batch_size, signal_len});

  // Apply filter to each row using slice + convolve
  int64_t out_len = signal_len + b.numel() - 1;

  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array x_cpu = (x_2d.place().kind() == DeviceKind::CPU) ? x_2d : x_2d.to(cpu);

  // Process each row
  std::vector<double> out_data(batch_size * out_len);
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;

  for (int64_t row = 0; row < batch_size; ++row) {
    // Extract row as 1D array
    Array row_arr =
        slice(x_cpu, {0}, {static_cast<int>(row)}, {static_cast<int>(row + 1)});
    row_arr = reshape(row_arr, {signal_len});
    if (row_arr.dtype() != work_dtype)
      row_arr = row_arr.to(work_dtype);

    // Convolve
    Array conv = convolve(row_arr, b_cpu, "full");

    // Copy result
    if (work_dtype == DType::F64) {
      const double *cd = conv.data<double>();
      for (int64_t i = 0; i < out_len; ++i)
        out_data[row * out_len + i] = cd[i];
    } else {
      const float *cd = conv.data<float>();
      for (int64_t i = 0; i < out_len; ++i)
        out_data[row * out_len + i] = cd[i];
    }
  }

  // Reshape back
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (i == ax)
      out_shape.push_back(out_len);
    else
      out_shape.push_back(x.shape().dim(i));
  }

  Array result = to_array(out_data, Shape(out_shape), DType::F64, cpu);

  // Move axis back if needed
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = permute(result, inv_perm);
  }

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// lfilter — IIR/FIR digital filter (sequential, requires backend kernel)
// ============================================================================

Array lfilter(const Array &b, const Array &a, const Array &x, int axis) {
  INS_CHECK(b.defined() && a.defined() && x.defined(),
            "lfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1 && a.shape().ndim() == 1,
            "lfilter: b and a must be 1D");

  // FIR case (a has only one element)
  if (a.numel() == 1) {
    return firfilter(b, x, axis);
  }

  // IIR case: direct-form II transpose
  // This is inherently sequential — process on CPU
  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array a_cpu = (a.place().kind() == DeviceKind::CPU) ? a : a.to(cpu);
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (b_cpu.dtype() != work_dtype)
    b_cpu = b_cpu.to(work_dtype);
  if (a_cpu.dtype() != work_dtype)
    a_cpu = a_cpu.to(work_dtype);
  if (x_cpu.dtype() != work_dtype)
    x_cpu = x_cpu.to(work_dtype);

  int64_t nb = b_cpu.numel();
  int64_t na = a_cpu.numel();

  // Normalize coefficients by a[0]
  double a0;
  {
    Array a0_arr = slice(a_cpu, {0}, {0}, {1});
    a0 = (work_dtype == DType::F64) ? a0_arr.item<double>()
                                    : static_cast<double>(a0_arr.item<float>());
  }
  INS_CHECK(std::abs(a0) > 1e-15, "lfilter: a[0] must not be zero");

  std::vector<double> b_norm(nb), a_norm(na);
  if (work_dtype == DType::F64) {
    const double *bp = b_cpu.data<double>();
    const double *ap = a_cpu.data<double>();
    for (int64_t i = 0; i < nb; ++i)
      b_norm[i] = bp[i] / a0;
    for (int64_t i = 0; i < na; ++i)
      a_norm[i] = ap[i] / a0;
  } else {
    const float *bp = b_cpu.data<float>();
    const float *ap = a_cpu.data<float>();
    for (int64_t i = 0; i < nb; ++i)
      b_norm[i] = bp[i] / a0;
    for (int64_t i = 0; i < na; ++i)
      a_norm[i] = ap[i] / a0;
  }

  int ndim = x_cpu.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;
  int64_t n = x_cpu.shape().dim(ax);
  int64_t batch = x_cpu.numel() / n;

  // Move axis to last
  Array x_moved = x_cpu;
  if (ax != ndim - 1) {
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    x_moved = permute(x_moved, perm);
  }

  // Direct-form II transpose IIR filter
  std::vector<double> out_data(batch * n);
  int64_t nmax = std::max(nb, na);

  if (work_dtype == DType::F64) {
    const double *x_ptr = x_moved.data<double>();
    for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
      const double *xi = x_ptr + b_idx * n;
      double *yi = out_data.data() + b_idx * n;
      std::vector<double> z(nmax - 1, 0.0);

      for (int64_t i = 0; i < n; ++i) {
        double val = xi[i] + z[0];
        yi[i] = val;
        for (int64_t j = 0; j < nmax - 2; ++j) {
          double bj1 = (j + 1 < nb) ? b_norm[j + 1] : 0.0;
          double aj1 = (j + 1 < na) ? a_norm[j + 1] : 0.0;
          z[j] = z[j + 1] + bj1 * xi[i] - aj1 * val;
        }
        if (nmax >= 2) {
          double bn = (nmax - 1 < nb) ? b_norm[nmax - 1] : 0.0;
          double an = (nmax - 1 < na) ? a_norm[nmax - 1] : 0.0;
          z[nmax - 2] = bn * xi[i] - an * val;
        }
      }
    }
  } else {
    const float *x_ptr = x_moved.data<float>();
    for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
      const float *xi = x_ptr + b_idx * n;
      double *yi = out_data.data() + b_idx * n;
      std::vector<double> z(nmax - 1, 0.0);

      for (int64_t i = 0; i < n; ++i) {
        double val = static_cast<double>(xi[i]) + z[0];
        yi[i] = val;
        for (int64_t j = 0; j < nmax - 2; ++j) {
          double bj1 = (j + 1 < nb) ? b_norm[j + 1] : 0.0;
          double aj1 = (j + 1 < na) ? a_norm[j + 1] : 0.0;
          z[j] = z[j + 1] + bj1 * xi[i] - aj1 * val;
        }
        if (nmax >= 2) {
          double bn = (nmax - 1 < nb) ? b_norm[nmax - 1] : 0.0;
          double an = (nmax - 1 < na) ? a_norm[nmax - 1] : 0.0;
          z[nmax - 2] = bn * xi[i] - an * val;
        }
      }
    }
  }

  // Reshape back
  std::vector<int64_t> out_shape;
  for (int i = 0; i < ndim; ++i) {
    out_shape.push_back((i == ax) ? n : x_cpu.shape().dim(i));
  }

  Array result = to_array(out_data, Shape(out_shape), DType::F64, cpu);
  if (ax != ndim - 1) {
    std::vector<int> inv_perm(ndim);
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = permute(result, inv_perm);
  }

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// lfilter_zi
// ============================================================================

Array lfilter_zi(const Array &b, const Array &a) {
  INS_CHECK(b.defined() && a.defined(), "lfilter_zi: inputs are undefined");

  Place cpu = CPUPlace();
  DType work_dtype = (b.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array b_cpu = b.dtype() == work_dtype ? b : b.to(work_dtype);
  Array a_cpu = a.dtype() == work_dtype ? a : a.to(work_dtype);
  if (b_cpu.place().kind() != DeviceKind::CPU)
    b_cpu = b_cpu.to(cpu);
  if (a_cpu.place().kind() != DeviceKind::CPU)
    a_cpu = a_cpu.to(cpu);

  int64_t nb = b_cpu.numel();
  int64_t na = a_cpu.numel();
  int64_t n = std::max(nb, na);

  // Normalize by a[0] using item()
  double a0;
  {
    Array a0_arr = slice(a_cpu, {0}, {0}, {1});
    a0 = (work_dtype == DType::F64) ? a0_arr.item<double>()
                                    : static_cast<double>(a0_arr.item<float>());
  }

  std::vector<double> b_norm(nb), a_norm(na);
  if (work_dtype == DType::F64) {
    const double *bp = b_cpu.data<double>();
    const double *ap = a_cpu.data<double>();
    for (int64_t i = 0; i < nb; ++i)
      b_norm[i] = bp[i] / a0;
    for (int64_t i = 0; i < na; ++i)
      a_norm[i] = ap[i] / a0;
  } else {
    const float *bp = b_cpu.data<float>();
    const float *ap = a_cpu.data<float>();
    for (int64_t i = 0; i < nb; ++i)
      b_norm[i] = bp[i] / a0;
    for (int64_t i = 0; i < na; ++i)
      a_norm[i] = ap[i] / a0;
  }

  int64_t m = n - 1;
  if (m == 0) {
    return to_array(std::vector<double>{}, work_dtype, cpu);
  }

  // Build companion matrix A (m x m)
  std::vector<double> A_data(m * m, 0.0);
  for (int64_t i = 0; i < m; ++i) {
    double ai1 = (i + 1 < na) ? a_norm[i + 1] : 0.0;
    A_data[i * m + 0] = -ai1;
    if (i + 1 < m) {
      A_data[i * m + (i + 1)] = 1.0;
    }
  }

  // Build B vector: b[1:] - a[1:] * b[0]
  std::vector<double> B_vec(m);
  for (int64_t i = 0; i < m; ++i) {
    double bi1 = (i + 1 < nb) ? b_norm[i + 1] : 0.0;
    double ai1 = (i + 1 < na) ? a_norm[i + 1] : 0.0;
    B_vec[i] = bi1 - ai1 * b_norm[0];
  }

  // Solve (I - A) * zi = B
  std::vector<double> IA_data(m * m);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < m; ++j) {
      IA_data[i * m + j] = (i == j ? 1.0 : 0.0) - A_data[i * m + j];
    }
  }

  Array IA = to_array(IA_data, Shape({m, m}), work_dtype, cpu);
  Array B_arr = to_array(B_vec, Shape({m, 1}), work_dtype, cpu);
  Array zi = solve(IA, B_arr);
  zi = reshape(zi, {m});

  return zi;
}

// ============================================================================
// filtfilt — zero-phase filtering
// ============================================================================

Array filtfilt(const Array &b, const Array &a, const Array &x, int axis) {
  INS_CHECK(b.defined() && a.defined() && x.defined(),
            "filtfilt: inputs are undefined");

  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;

  Array y = lfilter(b, a, x, axis);

  int64_t orig_len = x.shape().dim(ax);
  if (y.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y = slice(y, {ax}, {start}, {start + orig_len});
  }

  Array y_rev = flip(y, ax);
  y_rev = lfilter(b, a, y_rev, ax);

  if (y_rev.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y_rev = slice(y_rev, {ax}, {start}, {start + orig_len});
  }

  return flip(y_rev, ax);
}

// ============================================================================
// decimate — downsample with anti-aliasing
// ============================================================================

Array decimate(const Array &x, int64_t q, int axis, bool zero_phase) {
  INS_CHECK(x.defined(), "decimate: input is undefined");
  INS_CHECK(q >= 1, "decimate: q must be >= 1");
  if (q == 1)
    return x.copy();

  int64_t n = 10 * q + 1;
  double cutoff = 1.0 / q;
  Array h = firwin(n, {cutoff}, "hamming", "lowpass", true);
  Array a = to_array(std::vector<double>{1.0});

  Array filtered =
      zero_phase ? filtfilt(h, a, x, axis) : lfilter(h, a, x, axis);

  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;

  return slice(filtered, {ax}, {0}, {x.shape().dim(ax)}, {static_cast<int>(q)});
}

// ============================================================================
// resample — FFT-based resampling
// ============================================================================

Array resample(const Array &x, int64_t num, int axis) {
  INS_CHECK(x.defined(), "resample: input is undefined");
  INS_CHECK(num >= 1, "resample: num must be >= 1");

  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;
  int64_t n = x.shape().dim(ax);

  if (num == n)
    return x.copy();

  Place cpu = CPUPlace();
  DType cdtype = (x.dtype() == DType::F32) ? DType::C32 : DType::C64;

  // For 1D case
  Array Xf = fft::fft(x, n);

  // Build new spectrum with zero-padding or truncation
  std::vector<std::complex<double>> new_X(num, {0.0, 0.0});

  // Extract spectrum data
  Array Xf_cpu = (Xf.place().kind() == DeviceKind::CPU) ? Xf : Xf.to(cpu);
  const std::complex<double> *X_data =
      reinterpret_cast<const std::complex<double> *>(Xf_cpu.data<char>());

  if (num > n) {
    // Zero-pad in frequency domain
    int64_t half_n = n / 2;
    new_X[0] = X_data[0];
    for (int64_t i = 1; i <= half_n; ++i) {
      new_X[i] = X_data[i];
    }
    if (n % 2 == 0) {
      new_X[num - half_n] = X_data[half_n];
    }
    for (int64_t i = 1; i < half_n; ++i) {
      new_X[num - i] = X_data[n - i];
    }
  } else {
    // Truncate in frequency domain
    int64_t half_num = num / 2;
    new_X[0] = X_data[0];
    for (int64_t i = 1; i <= half_num; ++i) {
      new_X[i] = X_data[i];
    }
    for (int64_t i = 1; i < half_num; ++i) {
      new_X[num - i] = X_data[n - i];
    }
    if (num % 2 == 0) {
      new_X[half_num] = X_data[half_num];
    }
  }

  Array new_X_arr = to_array(new_X, cdtype, cpu);
  Array result = fft::ifft(new_X_arr, num);
  result = mul(result, full(result.shape(), static_cast<double>(num) / n,
                            result.dtype()));

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// resample_poly — polyphase resampling
// ============================================================================

Array resample_poly(const Array &x, int64_t up, int64_t down, int axis) {
  INS_CHECK(x.defined(), "resample_poly: input is undefined");
  INS_CHECK(up >= 1 && down >= 1, "resample_poly: up and down must be >= 1");

  if (up == 1 && down == 1)
    return x.copy();

  // Design anti-aliasing filter
  int64_t max_rate = std::max(up, down);
  double f_c = 1.0 / max_rate;
  int64_t n = 10 * max_rate + 1;
  Array h = firwin(n, {f_c}, "hamming", "lowpass", true);

  Place cpu = CPUPlace();
  Array h_up =
      mul(h, full(h.shape(), static_cast<double>(up), DType::F64, cpu));

  int64_t input_len = x.numel();
  int64_t n_out = static_cast<int64_t>(
      std::ceil(static_cast<double>(input_len) * up / down));

  // Upsample by inserting zeros using scatter
  int64_t upsampled_len = input_len * up;
  Array up_arr = zeros({upsampled_len}, x.dtype(), cpu);

  // Set every up-th element to x[i] * up
  Array indices = arange(0.0, static_cast<double>(upsampled_len),
                         static_cast<double>(up), DType::I64, cpu);
  Array x_scaled =
      mul(x, full(x.shape(), static_cast<double>(up), x.dtype(), cpu));
  if (x_scaled.place().kind() != DeviceKind::CPU)
    x_scaled = x_scaled.to(cpu);
  put(up_arr, indices, x_scaled);

  // Apply FIR filter
  DType work_dtype = (x.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (up_arr.dtype() != work_dtype)
    up_arr = up_arr.to(work_dtype);
  if (h_up.dtype() != work_dtype)
    h_up = h_up.to(work_dtype);
  Array filtered = convolve(up_arr, h_up, "full");

  // Downsample by taking every down-th sample
  int64_t offset = (h_up.numel() - 1) / 2;
  Array result = slice(filtered, {0}, {static_cast<int>(offset)},
                       {static_cast<int>(offset + n_out * down)},
                       {static_cast<int>(down)});

  if (result.dtype() != x.dtype())
    result = result.to(x.dtype());
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// freq_shift
// ============================================================================

Array freq_shift(const Array &x, double freq, double fs) {
  INS_CHECK(x.defined(), "freq_shift: input is undefined");
  INS_CHECK(fs > 0, "freq_shift: fs must be > 0");

  int64_t n = x.numel();
  Place cpu = CPUPlace();

  // Build complex exponential: exp(j*2*pi*freq/fs * [0,1,...,n-1])
  Array t = arange(0.0, static_cast<double>(n), 1.0, DType::F64, cpu);
  double phase_inc = 2.0 * M_PI * freq / fs;
  Array phase = mul(t, full(t.shape(), phase_inc, DType::F64, cpu));

  // cos(phase) + j*sin(phase)
  Array cos_val = cos(phase);
  Array sin_val = sin(phase);
  Array exp_arr = to_complex(cos_val, sin_val);

  // Convert x to complex if needed
  Array x_cpx;
  if (x.dtype() == DType::C64 || x.dtype() == DType::C32) {
    x_cpx = x;
  } else {
    x_cpx = to_complex(x);
  }

  // Element-wise multiply
  if (x_cpx.place().kind() != DeviceKind::CPU) {
    exp_arr = exp_arr.to(x_cpx.place());
  }

  Array result = mul(x_cpx, exp_arr);
  return result;
}

// ============================================================================
// wiener
// ============================================================================

Array wiener(const Array &im, const std::vector<int64_t> &mysize,
             double noise) {
  INS_CHECK(im.defined(), "wiener: input is undefined");

  int ndim = im.shape().ndim();

  std::vector<int64_t> sz = mysize;
  if (sz.empty()) {
    sz.assign(ndim, 3);
  }
  INS_CHECK(static_cast<int>(sz.size()) == ndim,
            "wiener: mysize must have same length as ndim");

  Place cpu = CPUPlace();
  DType work_dtype = (im.dtype() == DType::F32) ? DType::F32 : DType::F64;
  Array im_work = im.dtype() == work_dtype ? im : im.to(work_dtype);

  // Compute local mean using separable convolution
  Array local_mean = im_work;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, work_dtype, cpu);
    local_mean = convolve(local_mean, k_arr, "same");
  }

  // Compute local variance: E[x^2] - E[x]^2
  Array im_sq = mul(im_work, im_work);
  Array local_sq_mean = im_sq;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, work_dtype, cpu);
    local_sq_mean = convolve(local_sq_mean, k_arr, "same");
  }

  Array local_var = sub(local_sq_mean, mul(local_mean, local_mean));

  // Estimate noise if not provided
  double noise_est = noise;
  if (noise_est < 0) {
    Array var_mean = mean(local_var);
    noise_est = (work_dtype == DType::F64)
                    ? var_mean.item<double>()
                    : static_cast<double>(var_mean.item<float>());
    if (noise_est < 0)
      noise_est = 0;
  }

  // Wiener filter: result = local_mean + max(0, local_var - noise) / local_var
  // * (im - local_mean)
  Array noise_arr = full(local_var.shape(), noise_est, work_dtype, cpu);
  Array var_minus_noise = sub(local_var, noise_arr);
  Array zero = zeros_like(local_var);
  Array numerator = maximum(var_minus_noise, zero);

  Array eps = full(local_var.shape(), 1e-10, work_dtype, cpu);
  Array denominator = maximum(local_var, eps);

  Array gain = div(numerator, denominator);
  Array residual = sub(im_work, local_mean);
  Array result = add(local_mean, mul(gain, residual));

  if (result.dtype() != im.dtype())
    result = result.to(im.dtype());
  if (im.place().kind() != DeviceKind::CPU)
    result = result.to(im.place());
  return result;
}

} // namespace signal
} // namespace ins
