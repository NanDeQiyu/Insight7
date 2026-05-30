// src/ops/signal/filtering.cpp
#include "insight/ops/signal/filtering.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
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

  // Work on CPU for complex multiplication
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  // FFT of input
  Array Xf = fft::fft(x_cpu, N);

  // Build frequency-domain Hilbert transform filter:
  // h[0] = 1, h[1..N/2-1] = 2, h[N/2] = 1 (if N even), h[N/2+1..N-1] = 0
  // Multiply Xf by h element-wise (manual complex * real)
  std::vector<std::complex<double>> result_data(N);
  const std::complex<double> *X_data =
      reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

  result_data[0] = X_data[0]; // h[0] = 1
  for (int64_t i = 1; i < (N + 1) / 2; ++i) {
    result_data[i] = X_data[i] * 2.0; // h[i] = 2
  }
  if (N % 2 == 0) {
    result_data[N / 2] = X_data[N / 2]; // h[N/2] = 1
  }
  // h[N/2+1..N-1] = 0 (already zero-initialized)

  Array result_arr = to_array(result_data, DType::C64, cpu);
  Array result = fft::ifft(result_arr, N);

  // Transfer back if needed
  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
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

  // Build 2D frequency-domain filter as outer product of two 1D filters
  auto make_h = [](int64_t n) {
    std::vector<std::complex<double>> h(n, {0.0, 0.0});
    h[0] = {1.0, 0.0};
    for (int64_t i = 1; i < (n + 1) / 2; ++i)
      h[i] = {2.0, 0.0};
    if (n % 2 == 0)
      h[n / 2] = {1.0, 0.0};
    return h;
  };

  auto h_row = make_h(N);
  auto h_col = make_h(N);

  // Build 2D filter via outer product
  std::vector<std::complex<double>> h2d(N * N);
  for (int64_t i = 0; i < N; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      h2d[i * N + j] = h_row[i] * h_col[j];
    }
  }

  Place cpu = CPUPlace();
  Array h_arr = to_array(h2d, Shape({N, N}), DType::C64, cpu);
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
    // Subtract mean along axis
    Array m = mean(data, ax, true);
    return sub(data, m);
  }

  // Linear detrend: subtract best-fit line along axis
  int64_t n = data.shape().dim(ax);
  if (n <= 1)
    return data.copy();

  // Create index array [0, 1, 2, ..., n-1]
  Array t = arange(0.0, static_cast<double>(n), 1.0, DType::F64);

  // Compute linear regression along axis
  // slope = (n*sum(t*x) - sum(t)*sum(x)) / (n*sum(t^2) - (sum(t))^2)
  // intercept = mean(x) - slope * mean(t)

  // For multi-dimensional data, we need to compute per-slice
  // Simplified: reshape to 2D along the target axis, apply, reshape back

  // Mean of t
  double t_mean = (n - 1.0) / 2.0;
  double t_var = (n * n - 1.0) / 12.0; // var of [0..n-1]

  // Compute mean along axis
  Array x_mean = mean(data, ax, true);

  // Compute t*x along axis: broadcast t to match data shape
  // We need to reshape t so it broadcasts along axis `ax`
  // Strategy: create t with the right number of dimensions
  std::vector<int64_t> t_shape(ndim, 1);
  t_shape[ax] = n;
  t = reshape(t, Shape(t_shape));

  // Compute sum((t - t_mean) * (x - x_mean)) along axis
  Array t_centered = sub(t, full(Shape(t_shape), t_mean, DType::F64));
  Array x_centered = sub(data, x_mean);

  // Promote to F64 for computation
  DType work_dtype = (data.dtype() == DType::F32) ? DType::F32 : DType::F64;
  if (t_centered.dtype() != work_dtype)
    t_centered = t_centered.to(work_dtype);
  if (x_centered.dtype() != work_dtype)
    x_centered = x_centered.to(work_dtype);

  Array product = mul(t_centered, x_centered);
  Array sum_product = sum(product, ax, true);

  // slope = sum((t - t_mean) * (x - x_mean)) / (n * var(t))
  // = sum_product / (n * t_var)
  double denom = n * t_var;
  Array slope = div(sum_product, full(sum_product.shape(), denom, work_dtype));

  // trend = slope * (t - t_mean) + x_mean
  Array trend = add(mul(slope, t_centered), x_mean);

  Array result = sub(data, trend);
  if (result.dtype() != data.dtype()) {
    result = result.to(data.dtype());
  }
  return result;
}

// ============================================================================
// firfilter — FIR filtering via convolution along axis
// ============================================================================

namespace {

// Apply 1D FIR filter along a specific axis of a multi-dimensional array
Array firfilter_axis(const Array &b, const Array &x, int axis) {
  // For 1D input or axis=-1 with 1D, use simple convolution
  if (x.shape().ndim() == 1) {
    return convolve(x, b, "full");
  }

  // For multi-dimensional: apply convolve along each slice of the axis
  // Strategy: move target axis to last position, reshape to 2D, apply, reshape
  // back
  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;

  // Move target axis to last
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

  // Apply filter to each row
  int64_t out_len = signal_len + b.numel() - 1;
  std::vector<double> out_data(batch_size * out_len);

  // Get b coefficients
  const double *b_data = b.data<double>();

  for (int64_t row = 0; row < batch_size; ++row) {
    // Extract row
    std::vector<double> row_data(signal_len);
    const double *src = x_2d.data<double>() + row * signal_len;
    std::copy(src, src + signal_len, row_data.begin());

    // Convolve
    for (int64_t i = 0; i < out_len; ++i) {
      double val = 0.0;
      for (int64_t j = 0; j < b.numel(); ++j) {
        int64_t idx = i - j;
        if (idx >= 0 && idx < signal_len) {
          val += b_data[j] * row_data[idx];
        }
      }
      out_data[row * out_len + i] = val;
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

  Place cpu = CPUPlace();
  Array result = to_array(out_data, Shape(out_shape), DType::F64, cpu);

  // Move axis back if needed
  if (ax != ndim - 1) {
    // Inverse permutation
    std::vector<int> inv_perm(ndim);
    std::vector<int> perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    perm.erase(perm.begin() + ax);
    perm.push_back(ax);
    for (int i = 0; i < ndim; ++i)
      inv_perm[perm[i]] = i;
    result = permute(result, inv_perm);
  }

  return result;
}

} // anonymous namespace

Array firfilter(const Array &b, const Array &x, int axis) {
  INS_CHECK(b.defined() && x.defined(), "firfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1, "firfilter: b must be 1D");

  // Work on CPU for direct convolution (pointer access)
  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  Array result = firfilter_axis(b_cpu, x_cpu, axis);

  // Transfer back if needed
  if (x.place().kind() != DeviceKind::CPU) {
    result = result.to(x.place());
  }
  return result;
}

// ============================================================================
// lfilter — IIR/FIR digital filter
// ============================================================================

Array lfilter(const Array &b, const Array &a, const Array &x, int axis) {
  INS_CHECK(b.defined() && a.defined() && x.defined(),
            "lfilter: inputs are undefined");
  INS_CHECK(b.shape().ndim() == 1 && a.shape().ndim() == 1,
            "lfilter: b and a must be 1D");

  // Work on CPU (pointer access required for direct-form filter)
  Place cpu = CPUPlace();
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array a_cpu = (a.place().kind() == DeviceKind::CPU) ? a : a.to(cpu);
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  int64_t nb = b_cpu.numel();
  int64_t na = a_cpu.numel();

  // FIR case (a has only one element)
  if (na == 1) {
    Array result = firfilter_axis(b_cpu, x_cpu, axis);
    if (x.place().kind() != DeviceKind::CPU)
      result = result.to(x.place());
    return result;
  }

  // IIR case: direct-form II transpose
  // Normalize by a[0]
  const double *b_ptr = b_cpu.data<double>();
  const double *a_ptr = a_cpu.data<double>();

  INS_CHECK(std::abs(a_ptr[0]) > 1e-15, "lfilter: a[0] must not be zero");

  std::vector<double> b_norm(nb);
  std::vector<double> a_norm(na);
  for (int64_t i = 0; i < nb; ++i)
    b_norm[i] = b_ptr[i] / a_ptr[0];
  for (int64_t i = 0; i < na; ++i)
    a_norm[i] = a_ptr[i] / a_ptr[0];

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

  // Process each batch element
  std::vector<double> out_data(batch * n);
  const double *x_ptr = x_moved.data<double>();

  int64_t nmax = std::max(nb, na);
  for (int64_t b_idx = 0; b_idx < batch; ++b_idx) {
    const double *xi = x_ptr + b_idx * n;
    double *yi = out_data.data() + b_idx * n;

    // State vector
    std::vector<double> z(nmax - 1, 0.0);

    for (int64_t i = 0; i < n; ++i) {
      double val = xi[i] + z[0];
      yi[i] = val;

      // Update state
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
  Array b_cpu = (b.place().kind() == DeviceKind::CPU) ? b : b.to(cpu);
  Array a_cpu = (a.place().kind() == DeviceKind::CPU) ? a : a.to(cpu);

  const double *b_ptr = b_cpu.data<double>();
  const double *a_ptr = a_cpu.data<double>();

  int64_t nb = b_cpu.numel();
  int64_t na = a_cpu.numel();
  int64_t n = std::max(nb, na);

  // Normalize by a[0]
  std::vector<double> b_norm(nb);
  std::vector<double> a_norm(na);
  for (int64_t i = 0; i < nb; ++i)
    b_norm[i] = b_ptr[i] / a_ptr[0];
  for (int64_t i = 0; i < na; ++i)
    a_norm[i] = a_ptr[i] / a_ptr[0];

  // Build companion matrix A (n-1 x n-1)
  // A = [[-a[1], 1, 0, ...],
  //      [-a[2], 0, 1, ...],
  //      ...
  //      [-a[n-1], 0, 0, ...]]
  int64_t m = n - 1;
  if (m == 0) {
    return to_array(std::vector<double>{}, DType::F64, cpu);
  }

  // Build A matrix
  std::vector<double> A_data(m * m, 0.0);
  for (int64_t i = 0; i < m; ++i) {
    double ai1 = (i + 1 < na) ? a_norm[i + 1] : 0.0;
    A_data[i * m + 0] = -ai1; // first column
    if (i + 1 < m) {
      A_data[i * m + (i + 1)] = 1.0; // superdiagonal
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
  // I - A
  std::vector<double> IA_data(m * m);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < m; ++j) {
      IA_data[i * m + j] = (i == j ? 1.0 : 0.0) - A_data[i * m + j];
    }
  }

  // Use linalg::solve for (I-A) * zi = B
  Array IA = to_array(IA_data, Shape({m, m}), DType::F64, cpu);
  Array B_arr = to_array(B_vec, Shape({m, 1}), DType::F64, cpu);
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

  // Forward filter
  Array y = lfilter(b, a, x, axis);

  // Crop to original length (lfilter may extend via full convolution)
  int64_t orig_len = x.shape().dim(ax);
  if (y.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y = slice(y, {ax}, {start}, {start + orig_len});
  }

  // Reverse along axis, filter again, reverse back
  Array y_rev = flip(y, ax);
  y_rev = lfilter(b, a, y_rev, ax);

  // Crop again
  if (y_rev.shape().dim(ax) > orig_len) {
    int64_t nb = b.numel();
    int64_t start = nb - 1;
    y_rev = slice(y_rev, {ax}, {start}, {start + orig_len});
  }

  Array result = flip(y_rev, ax);
  return result;
}

// ============================================================================
// decimate — downsample with anti-aliasing
// ============================================================================

Array decimate(const Array &x, int64_t q, int axis, bool zero_phase) {
  INS_CHECK(x.defined(), "decimate: input is undefined");
  INS_CHECK(q >= 1, "decimate: q must be >= 1");
  if (q == 1)
    return x.copy();

  // Design anti-aliasing FIR filter
  int64_t n = 10 * q + 1; // filter length
  // Lowpass FIR with cutoff at 1/q (normalized)
  double cutoff = 1.0 / q;
  Array h = firwin(n, {cutoff}, "hamming", "lowpass", true);

  // Apply filter
  Array filtered;
  if (zero_phase) {
    filtered = filtfilt(h, to_array(std::vector<double>{1.0}), x, axis);
  } else {
    filtered = lfilter(h, to_array(std::vector<double>{1.0}), x, axis);
  }

  // Downsample by taking every q-th sample along axis
  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;

  Array downsampled =
      slice(filtered, {ax}, {0}, {x.shape().dim(ax)}, {static_cast<int>(q)});

  return downsampled;
}

// ============================================================================
// resample — FFT-based resampling
// ============================================================================

Array resample(const Array &x, int64_t num, int axis) {
  INS_CHECK(x.defined(), "resample: input is undefined");
  INS_CHECK(num >= 1, "resample: num must be >= 1");

  Place cpu = CPUPlace();
  int ndim = x.shape().ndim();
  int ax = (axis < 0) ? axis + ndim : axis;
  int64_t n = x.shape().dim(ax);

  if (num == n)
    return x.copy();

  // Work on CPU
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);

  // For 1D case, use FFT resampling
  if (ndim == 1) {
    Array Xf = fft::fft(x_cpu, n);

    const std::complex<double> *X_data =
        reinterpret_cast<const std::complex<double> *>(Xf.data<char>());

    if (num > n) {
      // Zero-pad in frequency domain
      int64_t half_n = n / 2;
      std::vector<std::complex<double>> new_X(num, {0.0, 0.0});
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

      Array new_X_arr = to_array(new_X, DType::C64, cpu);
      Array result = fft::ifft(new_X_arr, num);
      // Scale
      result = mul(result, full(result.shape(), static_cast<double>(num) / n,
                                result.dtype()));
      if (result.dtype() != x.dtype())
        result = result.to(x.dtype());
      if (x.place().kind() != DeviceKind::CPU)
        result = result.to(x.place());
      return result;
    } else {
      // Truncate in frequency domain
      int64_t half_num = num / 2;
      std::vector<std::complex<double>> new_X(num, {0.0, 0.0});
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

      Array new_X_arr = to_array(new_X, DType::C64, cpu);
      Array result = fft::ifft(new_X_arr, num);
      result = mul(result, full(result.shape(), static_cast<double>(num) / n,
                                result.dtype()));
      if (result.dtype() != x.dtype())
        result = result.to(x.dtype());
      if (x.place().kind() != DeviceKind::CPU)
        result = result.to(x.place());
      return result;
    }
  }

  // Multi-dimensional: simplified - just handle 1D for now
  INS_CHECK(false, "resample: multi-dimensional not yet supported");
  return Array();
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
  // Filter length: 10 * max_rate + 1
  int64_t n = 10 * max_rate + 1;
  Array h = firwin(n, {f_c}, "hamming", "lowpass", true);

  // Scale filter by upsample factor
  Place cpu = CPUPlace();
  Array h_up =
      mul(h, full(h.shape(), static_cast<double>(up), DType::F64, cpu));

  // Zero-pad input
  int64_t input_len = x.numel(); // Assuming 1D for now
  int64_t n_out = static_cast<int64_t>(
      std::ceil(static_cast<double>(input_len) * up / down));

  // Upsample by inserting zeros
  int64_t upsampled_len = input_len * up;
  std::vector<double> up_data(upsampled_len, 0.0);
  const double *x_data = x.data<double>();
  for (int64_t i = 0; i < input_len; ++i) {
    up_data[i * up] = x_data[i] * up;
  }

  // Apply FIR filter
  Array up_arr = to_array(up_data, DType::F64, cpu);
  Array filtered = convolve(up_arr, h_up, "full");

  // Downsample by taking every down-th sample
  // The output starts at offset (n-1) to account for filter delay
  int64_t offset = (h_up.numel() - 1) / 2;
  std::vector<double> out_data(n_out);
  const double *f_data = filtered.data<double>();
  for (int64_t i = 0; i < n_out; ++i) {
    int64_t idx = offset + i * down;
    if (idx >= 0 && idx < filtered.numel()) {
      out_data[i] = f_data[idx];
    }
  }

  Array result = to_array(out_data, DType::F64, cpu);
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

  // Compute complex exponential on CPU
  double phase_inc = 2.0 * M_PI * freq / fs;
  std::vector<std::complex<double>> exp_data(n);
  for (int64_t i = 0; i < n; ++i) {
    double phase = phase_inc * i;
    exp_data[i] = {std::cos(phase), std::sin(phase)};
  }

  // Convert x to complex on CPU
  Array x_cpu = (x.place().kind() == DeviceKind::CPU) ? x : x.to(cpu);
  std::vector<std::complex<double>> cpx_data(n);
  if (x_cpu.dtype() == DType::C64) {
    const std::complex<double> *xd =
        reinterpret_cast<const std::complex<double> *>(x_cpu.data<char>());
    for (int64_t i = 0; i < n; ++i)
      cpx_data[i] = xd[i];
  } else if (x_cpu.dtype() == DType::C32) {
    const std::complex<float> *xd =
        reinterpret_cast<const std::complex<float> *>(x_cpu.data<char>());
    for (int64_t i = 0; i < n; ++i)
      cpx_data[i] = xd[i];
  } else if (x_cpu.dtype() == DType::F64) {
    const double *xd = x_cpu.data<double>();
    for (int64_t i = 0; i < n; ++i)
      cpx_data[i] = {xd[i], 0.0};
  } else {
    const float *xd = x_cpu.data<float>();
    for (int64_t i = 0; i < n; ++i)
      cpx_data[i] = {xd[i], 0.0};
  }

  // Multiply element-wise
  std::vector<std::complex<double>> result_data(n);
  for (int64_t i = 0; i < n; ++i) {
    result_data[i] = cpx_data[i] * exp_data[i];
  }

  Array result = to_array(result_data, DType::C64, cpu);
  if (x.place().kind() != DeviceKind::CPU)
    result = result.to(x.place());
  return result;
}

// ============================================================================
// wiener
// ============================================================================

Array wiener(const Array &im, const std::vector<int64_t> &mysize,
             double noise) {
  INS_CHECK(im.defined(), "wiener: input is undefined");

  Place cpu = CPUPlace();
  int ndim = im.shape().ndim();

  // Default mysize: 3 for each dimension
  std::vector<int64_t> sz = mysize;
  if (sz.empty()) {
    sz.assign(ndim, 3);
  }
  INS_CHECK(static_cast<int>(sz.size()) == ndim,
            "wiener: mysize must have same length as ndim");

  // Compute local mean using uniform filter (approximate with convolution)
  // Create averaging kernel
  int64_t kernel_size = 1;
  for (auto s : sz)
    kernel_size *= s;
  double kernel_val = 1.0 / kernel_size;

  // For simplicity, use 1D case with symmetric padding
  // General N-D implementation using separable convolution
  Array im_work = im;
  if (im_work.dtype() != DType::F64)
    im_work = im_work.to(DType::F64);
  if (im_work.place().kind() != DeviceKind::CPU)
    im_work = im_work.to(cpu);

  // Create 1D averaging kernel for each dimension
  Array local_mean = im_work;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, DType::F64, cpu);

    // Pad the signal symmetrically
    int64_t pad = k_size / 2;
    // Use convolve with "same" mode (centered)
    // For each slice along dimension d, convolve with kernel
    // Simplified: use direct computation for 1D
    if (ndim == 1) {
      local_mean = ins::convolve(im_work, k_arr, "same");
    } else {
      // For N-D, apply separable 1D convolution along each axis
      // This is a simplified version
      local_mean = ins::convolve(im_work, k_arr, "same");
    }
  }

  // Compute local variance: E[x^2] - E[x]^2
  Array im_sq = mul(im_work, im_work);
  Array local_sq_mean = im_sq;
  for (int d = 0; d < ndim; ++d) {
    int64_t k_size = sz[d];
    std::vector<double> kernel(k_size, 1.0 / k_size);
    Array k_arr = to_array(kernel, DType::F64, cpu);
    local_sq_mean = ins::convolve(local_sq_mean, k_arr, "same");
  }

  Array local_var = sub(local_sq_mean, mul(local_mean, local_mean));

  // Estimate noise if not provided
  double noise_est = noise;
  if (noise_est < 0) {
    Array var_mean = mean(local_var);
    noise_est = var_mean.data<double>()[0];
    if (noise_est < 0)
      noise_est = 0;
  }

  // Wiener filter: result = local_mean + max(0, local_var - noise) / local_var
  // * (im - local_mean)
  Array noise_arr = full(local_var.shape(), noise_est, DType::F64, cpu);
  Array var_minus_noise = sub(local_var, noise_arr);
  Array zero = zeros_like(local_var);
  Array numerator = maximum(var_minus_noise, zero);

  // Avoid division by zero
  Array eps = full(local_var.shape(), 1e-10, DType::F64, cpu);
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
