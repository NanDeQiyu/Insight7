// bindings/julia/insight_julia_capi.cpp
//
// C ABI wrapper functions for Julia bindings.
// Julia calls these via ccall(). Each function wraps the C++ ins:: API.
// Compiled into libinsight_julia.so.

#include "insight/core/array.h"
#include "insight/core/dtype.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/init.h"
#include "insight/ops/cast.h"
#include "insight/ops/complex.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/random.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include "insight/ops/unary.h"
#ifdef INSIGHT_USE_MATPLOT
#include "insight/ops/plot.h"
#endif

#include <cstring>
#include <string>
#include <vector>

using namespace ins;

// All exported functions use C linkage for Julia ccall().
extern "C" {

// ============================================================================
// Initialization
// ============================================================================

void insight_jl_init_cpu() {
  if (!ins::is_initialized()) {
    // Try to load backends — dlopen will search LD_LIBRARY_PATH
    try {
      ins::init({"cpu"});
    } catch (...) {
      // If dynamic loading fails, the user needs to set LD_LIBRARY_PATH
      fprintf(stderr,
              "[insight] Warning: CPU backend not found. "
              "Set LD_LIBRARY_PATH to include the backend directory.\n");
    }
  }
}

// ============================================================================
// Device information
// ============================================================================

void insight_jl_device_name(int32_t device_id, char *buf, size_t buf_size) {
  std::string name = device_name(DeviceKind::GPU, device_id);
  std::strncpy(buf, name.c_str(), buf_size - 1);
  buf[buf_size - 1] = '\0';
}

int32_t insight_jl_cuda_version() { return cuda_version(); }
int32_t insight_jl_driver_version() { return driver_version(); }
int32_t insight_jl_compute_capability(int32_t device_id) {
  return compute_capability(device_id);
}

void insight_jl_device_memory(int32_t device_id, size_t *total, size_t *free) {
  auto info = device_memory(device_id);
  *total = info.total;
  *free = info.free;
}

int32_t insight_jl_gpu_count() {
  return static_cast<int>(device_count(DeviceKind::GPU));
}

// ============================================================================
// Array lifecycle
// ============================================================================

// Create a new array. Returns heap-allocated Array*. Caller must free with
// insight_jl_array_free().
Array *insight_jl_zeros(const int64_t *dims, int32_t ndim, int32_t dtype,
                        int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(zeros(shape, dt, place));
}

Array *insight_jl_ones(const int64_t *dims, int32_t ndim, int32_t dtype,
                       int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(ones(shape, dt, place));
}

Array *insight_jl_full(const int64_t *dims, int32_t ndim, double fill_value,
                       int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(full(shape, fill_value, dt, place));
}

// Create array from raw data (Julia-owned memory, copied into Insight).
Array *insight_jl_from_data(const void *data, const int64_t *dims, int32_t ndim,
                            int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  Array *arr = new Array(shape, dt, CPUPlace());
  std::memcpy(arr->data(), data, shape.numel() * dtype_size(dt));
  if (device_type == 1) {
    *arr = arr->to(GPUPlace(0));
  }
  return arr;
}

// Copy array data to a Julia-owned buffer (must be contiguous, CPU).
void insight_jl_to_data(const Array *arr, void *dst) {
  Array cpu = arr->contiguous().to(CPUPlace());
  std::memcpy(dst, cpu.data(), cpu.nbytes());
}

void insight_jl_array_free(Array *arr) { delete arr; }

// Metadata queries
int32_t insight_jl_ndim(const Array *arr) { return arr->shape().ndim(); }
int64_t insight_jl_numel(const Array *arr) { return arr->numel(); }
int32_t insight_jl_dtype(const Array *arr) {
  return static_cast<int32_t>(arr->dtype());
}
int32_t insight_jl_device_type(const Array *arr) {
  return arr->place().is_gpu() ? 1 : 0;
}
void insight_jl_shape(const Array *arr, int64_t *dims_out) {
  for (int i = 0; i < arr->shape().ndim(); i++) {
    dims_out[i] = arr->shape().dim(i);
  }
}

// ============================================================================
// Creation
// ============================================================================

Array *insight_jl_arange(double start, double end, double step, int32_t dtype,
                         int32_t device_type) {
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(arange(start, end, step, dt, place));
}

Array *insight_jl_linspace(double start, double stop, int64_t num,
                           int32_t dtype, int32_t device_type) {
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(linspace(start, stop, num, dt, place));
}

Array *insight_jl_eye(int64_t n, int64_t m, int32_t dtype,
                      int32_t device_type) {
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(eye(n, m, 0, dt, place));
}

// ============================================================================
// Elementwise (binary)
// ============================================================================

#define JL_BINARY_OP(name, cpp_func)                                           \
  Array *insight_jl_##name(const Array *a, const Array *b) {                   \
    return new Array(cpp_func(*a, *b));                                        \
  }

JL_BINARY_OP(add, add)
JL_BINARY_OP(sub, sub)
JL_BINARY_OP(mul, mul)
JL_BINARY_OP(div, div)
JL_BINARY_OP(pow, pow)
JL_BINARY_OP(mod, mod)
JL_BINARY_OP(maximum, maximum)
JL_BINARY_OP(minimum, minimum)
JL_BINARY_OP(equal, equal)
JL_BINARY_OP(not_equal, not_equal)
JL_BINARY_OP(greater, greater)
JL_BINARY_OP(less, less)
JL_BINARY_OP(greater_equal, greater_equal)
JL_BINARY_OP(less_equal, less_equal)

// ============================================================================
// Unary
// ============================================================================

#define JL_UNARY_OP(name, cpp_func)                                            \
  Array *insight_jl_##name(const Array *x) { return new Array(cpp_func(*x)); }

JL_UNARY_OP(negative, negative)
JL_UNARY_OP(abs, ins::abs)
JL_UNARY_OP(square, square)
JL_UNARY_OP(sqrt, sqrt)
JL_UNARY_OP(exp, exp)
JL_UNARY_OP(log, log)
JL_UNARY_OP(log2, log2)
JL_UNARY_OP(log10, log10)
JL_UNARY_OP(sin, sin)
JL_UNARY_OP(cos, cos)
JL_UNARY_OP(tan, tan)
JL_UNARY_OP(asin, asin)
JL_UNARY_OP(acos, acos)
JL_UNARY_OP(atan, atan)
JL_UNARY_OP(sinh, sinh)
JL_UNARY_OP(cosh, cosh)
JL_UNARY_OP(tanh, tanh)
JL_UNARY_OP(floor, floor)
JL_UNARY_OP(ceil, ceil)
JL_UNARY_OP(round, rint)
JL_UNARY_OP(sign, sign)

// ============================================================================
// Reduction
// ============================================================================

#define JL_REDUCTION(name, cpp_func)                                           \
  Array *insight_jl_##name(const Array *x, int32_t has_axis, int32_t axis,     \
                           int32_t keepdims) {                                 \
    std::optional<int> ax =                                                    \
        has_axis ? std::optional<int>(axis) : std::nullopt;                    \
    return new Array(cpp_func(*x, ax, keepdims != 0));                         \
  }

JL_REDUCTION(sum, sum)
JL_REDUCTION(mean, mean)
JL_REDUCTION(max, max)
JL_REDUCTION(min, min)
JL_REDUCTION(prod, prod)
JL_REDUCTION(argmax, argmax)
JL_REDUCTION(argmin, argmin)

// ============================================================================
// Linear Algebra
// ============================================================================

Array *insight_jl_matmul(const Array *a, const Array *b) {
  return new Array(matmul(*a, *b));
}
Array *insight_jl_dot(const Array *a, const Array *b) {
  return new Array(dot(*a, *b));
}
Array *insight_jl_outer(const Array *a, const Array *b) {
  return new Array(outer(*a, *b));
}
Array *insight_jl_det(const Array *x) { return new Array(det(*x)); }
Array *insight_jl_inv(const Array *x) { return new Array(inv(*x)); }
Array *insight_jl_trace(const Array *x) { return new Array(trace(*x)); }
Array *insight_jl_norm(const Array *x, double ord) {
  return new Array(norm(*x, ord));
}
Array *insight_jl_cholesky(const Array *x, int32_t lower) {
  return new Array(cholesky(*x, lower != 0));
}
Array *insight_jl_solve(const Array *a, const Array *b) {
  return new Array(solve(*a, *b));
}

// SVD returns U, S, Vt as three output arrays.
void insight_jl_svd(const Array *x, Array **U, Array **S, Array **Vt) {
  auto [u, s, vt] = svd(*x);
  *U = new Array(std::move(u));
  *S = new Array(std::move(s));
  *Vt = new Array(std::move(vt));
}

// QR returns Q, R.
void insight_jl_qr(const Array *x, Array **Q, Array **R) {
  auto [q, r] = qr(*x);
  *Q = new Array(std::move(q));
  *R = new Array(std::move(r));
}

// Eigh returns eigenvalues, eigenvectors.
void insight_jl_eigh(const Array *x, Array **vals, Array **vecs) {
  auto [v, w] = eigh(*x);
  *vals = new Array(std::move(v));
  *vecs = new Array(std::move(w));
}

// ============================================================================
// FFT
// ============================================================================

Array *insight_jl_fft(const Array *x, int32_t has_n, int64_t n) {
  return new Array(fft::fft(*x, has_n ? n : -1));
}
Array *insight_jl_ifft(const Array *x, int32_t has_n, int64_t n) {
  return new Array(fft::ifft(*x, has_n ? n : -1));
}
Array *insight_jl_rfft(const Array *x, int32_t has_n, int64_t n) {
  return new Array(fft::rfft(*x, has_n ? n : -1));
}
Array *insight_jl_fftfreq(int64_t n, double d) {
  return new Array(fft::fftfreq(n, d));
}

// ============================================================================
// Signal
// ============================================================================

Array *insight_jl_convolve(const Array *a, const Array *v, const char *mode) {
  return new Array(convolve(*a, *v, std::string(mode)));
}

Array *insight_jl_unwrap(const Array *p, int32_t axis, double discont,
                         double period) {
  return new Array(unwrap(*p, axis, discont, period));
}

Array *insight_jl_sinc(const Array *x) { return new Array(sinc(*x)); }

// --- Windows ---
Array *insight_jl_hann(int64_t M, int32_t sym) {
  return new Array(signal::hann(M, sym != 0));
}
Array *insight_jl_hamming(int64_t M, int32_t sym) {
  return new Array(signal::hamming(M, sym != 0));
}
Array *insight_jl_blackman(int64_t M, int32_t sym) {
  return new Array(signal::blackman(M, sym != 0));
}
Array *insight_jl_kaiser(int64_t M, double beta, int32_t sym) {
  return new Array(signal::kaiser(M, beta, sym != 0));
}
Array *insight_jl_gaussian(int64_t M, double std_val, int32_t sym) {
  return new Array(signal::gaussian(M, std_val, sym != 0));
}
Array *insight_jl_boxcar(int64_t M, int32_t sym) {
  return new Array(signal::boxcar(M, sym != 0));
}
Array *insight_jl_triang(int64_t M, int32_t sym) {
  return new Array(signal::triang(M, sym != 0));
}
Array *insight_jl_bartlett(int64_t M, int32_t sym) {
  return new Array(signal::bartlett(M, sym != 0));
}
Array *insight_jl_cosine_win(int64_t M, int32_t sym) {
  return new Array(signal::cosine(M, sym != 0));
}
Array *insight_jl_flattop(int64_t M, int32_t sym) {
  return new Array(signal::flattop(M, sym != 0));
}
Array *insight_jl_nuttall(int64_t M, int32_t sym) {
  return new Array(signal::nuttall(M, sym != 0));
}
Array *insight_jl_blackmanharris(int64_t M, int32_t sym) {
  return new Array(signal::blackmanharris(M, sym != 0));
}
Array *insight_jl_tukey(int64_t M, double alpha, int32_t sym) {
  return new Array(signal::tukey(M, alpha, sym != 0));
}
Array *insight_jl_chebwin(int64_t M, double at, int32_t sym) {
  return new Array(signal::chebwin(M, at, sym != 0));
}
Array *insight_jl_taylor(int64_t M, int64_t nbar, double sll, int32_t norm,
                         int32_t sym) {
  return new Array(signal::taylor(M, nbar, sll, norm != 0, sym != 0));
}
Array *insight_jl_general_hamming(int64_t M, double alpha, int32_t sym) {
  return new Array(signal::general_hamming(M, alpha, sym != 0));
}
Array *insight_jl_get_window(const char *window, int64_t Nx, int32_t fftbins) {
  return new Array(signal::get_window(std::string(window), Nx, fftbins != 0));
}

// --- Waveforms ---
Array *insight_jl_sawtooth(const Array *t, double width) {
  return new Array(signal::sawtooth(*t, width));
}
Array *insight_jl_square_wf(const Array *t, double duty) {
  return new Array(signal::square(*t, duty));
}
Array *insight_jl_chirp(const Array *t, double f0, double t1, double f1,
                        int32_t method, double phi, int32_t vertex_zero) {
  return new Array(signal::chirp(*t, f0, t1, f1,
                                 static_cast<signal::ChirpMethod>(method), phi,
                                 vertex_zero != 0));
}
Array *insight_jl_unit_impulse(const int64_t *dims, int32_t ndim, int64_t idx) {
  return new Array(signal::unit_impulse(
      Shape(std::vector<int64_t>(dims, dims + ndim)), idx));
}

// --- B-Splines ---
Array *insight_jl_gauss_spline(const Array *x, int32_t n) {
  return new Array(signal::gauss_spline(*x, n));
}
Array *insight_jl_cubic(const Array *x) { return new Array(signal::cubic(*x)); }
Array *insight_jl_quadratic(const Array *x) {
  return new Array(signal::quadratic(*x));
}

// --- Filter Design ---
double insight_jl_kaiser_beta(double a) { return signal::kaiser_beta(a); }
double insight_jl_kaiser_atten(int64_t numtaps, double width) {
  return signal::kaiser_atten(numtaps, width);
}
Array *insight_jl_firwin(int64_t numtaps, const double *cutoff,
                         int32_t cutoff_len, const char *window,
                         const char *pass_zero, int32_t scale) {
  std::vector<double> cv(cutoff, cutoff + cutoff_len);
  return new Array(signal::firwin(numtaps, cv, std::string(window),
                                  std::string(pass_zero), scale != 0));
}
Array *insight_jl_cmplx_sort(const Array *p) {
  return new Array(signal::cmplx_sort(*p));
}

// --- Convolution ---
Array *insight_jl_fftconvolve(const Array *in1, const Array *in2,
                              const char *mode) {
  return new Array(signal::fftconvolve(*in1, *in2, std::string(mode)));
}
Array *insight_jl_correlate(const Array *in1, const Array *in2,
                            const char *mode) {
  return new Array(signal::correlate(*in1, *in2, std::string(mode)));
}
Array *insight_jl_correlation_lags(int64_t in1_len, int64_t in2_len,
                                   const char *mode) {
  return new Array(
      signal::correlation_lags(in1_len, in2_len, std::string(mode)));
}

// --- Filtering ---
Array *insight_jl_hilbert(const Array *x, int64_t N) {
  return new Array(signal::hilbert(*x, N));
}
Array *insight_jl_detrend(const Array *data, int32_t axis, const char *type) {
  return new Array(signal::detrend(*data, axis, std::string(type)));
}
Array *insight_jl_lfilter(const Array *b, const Array *a, const Array *x,
                          int32_t axis) {
  return new Array(signal::lfilter(*b, *a, *x, axis));
}
Array *insight_jl_filtfilt(const Array *b, const Array *a, const Array *x,
                           int32_t axis) {
  return new Array(signal::filtfilt(*b, *a, *x, axis));
}
Array *insight_jl_decimate(const Array *x, int64_t q, int32_t axis,
                           int32_t zero_phase) {
  return new Array(signal::decimate(*x, q, axis, zero_phase != 0));
}
Array *insight_jl_resample(const Array *x, int64_t num, int32_t axis) {
  return new Array(signal::resample(*x, num, axis));
}
Array *insight_jl_freq_shift(const Array *x, double freq, double fs) {
  return new Array(signal::freq_shift(*x, freq, fs));
}

// --- Spectral Analysis ---
// welch returns (f, Pxx) — two arrays
void insight_jl_welch(const Array *x, double fs, const char *window,
                      int64_t nperseg, int64_t noverlap, int64_t nfft,
                      const char *detrend_str, int32_t return_onesided,
                      const char *scaling, Array **f_out, Array **Pxx_out) {
  auto res = signal::welch(*x, fs, std::string(window), nperseg, noverlap, nfft,
                           std::string(detrend_str), return_onesided != 0,
                           std::string(scaling));
  *f_out = new Array(std::move(res.f));
  *Pxx_out = new Array(std::move(res.Pxx));
}

void insight_jl_periodogram(const Array *x, double fs, const char *window,
                            int64_t nfft, const char *detrend_str,
                            int32_t return_onesided, const char *scaling,
                            Array **f_out, Array **Pxx_out) {
  auto res = signal::periodogram(*x, fs, std::string(window), nfft,
                                 std::string(detrend_str), return_onesided != 0,
                                 std::string(scaling));
  *f_out = new Array(std::move(res.f));
  *Pxx_out = new Array(std::move(res.Pxx));
}

// --- Wavelets ---
Array *insight_jl_morlet(int64_t M, double w, double s, int32_t complete) {
  return new Array(signal::morlet(M, w, s, complete != 0));
}
Array *insight_jl_ricker(int64_t points, double a) {
  return new Array(signal::ricker(points, a));
}

// --- Acoustics ---
Array *insight_jl_mel2hz(const Array *mels) {
  return new Array(signal::mel2hz(*mels));
}
Array *insight_jl_hz2mel(const Array *hz) {
  return new Array(signal::hz2mel(*hz));
}
Array *insight_jl_mel_frequencies(int64_t n_mels, double f_low, double f_high) {
  return new Array(signal::mel_frequencies(n_mels, f_low, f_high));
}
Array *insight_jl_hz2bark(const Array *hz) {
  return new Array(signal::hz2bark(*hz));
}
Array *insight_jl_bark2hz(const Array *bark) {
  return new Array(signal::bark2hz(*bark));
}

// ============================================================================
// Demod
// ============================================================================

Array *insight_jl_fm_demod(const Array *x, int32_t axis) {
  return new Array(signal::fm_demod(*x, axis));
}

// ============================================================================
// Peak Finding
// ============================================================================

// Returns number of extrema found; caller provides output buffer for indices
int64_t insight_jl_argrelmax(const Array *data, int32_t axis, int32_t order,
                             int64_t *out_indices) {
  auto result = signal::argrelmax(*data, axis, order, "clip");
  if (result.empty())
    return 0;
  int64_t n = result[0].numel();
  const int64_t *src = result[0].data<int64_t>();
  for (int64_t i = 0; i < n; ++i)
    out_indices[i] = src[i];
  return n;
}

int64_t insight_jl_argrelmin(const Array *data, int32_t axis, int32_t order,
                             int64_t *out_indices) {
  auto result = signal::argrelmin(*data, axis, order, "clip");
  if (result.empty())
    return 0;
  int64_t n = result[0].numel();
  const int64_t *src = result[0].data<int64_t>();
  for (int64_t i = 0; i < n; ++i)
    out_indices[i] = src[i];
  return n;
}

// ============================================================================
// Radar
// ============================================================================

Array *insight_jl_cfar_alpha(double pfa, int32_t N) {
  double alpha = signal::cfar_alpha(pfa, N);
  return new Array(alpha);
}

// ============================================================================
// Signal I/O
// ============================================================================

Array *insight_jl_read_bin(const char *file, int32_t dtype, int64_t num_samples,
                           int64_t offset) {
  return new Array(
      signal::read_bin(file, static_cast<DType>(dtype), num_samples, offset));
}

void insight_jl_write_bin(const char *file, const Array *data, int32_t append) {
  signal::write_bin(file, *data, append != 0);
}

Array *insight_jl_pack_bin(const Array *data) {
  return new Array(signal::pack_bin(*data));
}

Array *insight_jl_unpack_bin(const Array *binary, int32_t dtype,
                             const char *endianness) {
  return new Array(
      signal::unpack_bin(*binary, static_cast<DType>(dtype), endianness));
}

// ============================================================================
// Random
// ============================================================================

Array *insight_jl_rand(const int64_t *dims, int32_t ndim, int32_t dtype,
                       int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(rand(shape, dt, place));
}

Array *insight_jl_randn(const int64_t *dims, int32_t ndim, int32_t dtype,
                        int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  DType dt = static_cast<DType>(dtype);
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(randn(shape, dt, place));
}

// ============================================================================
// Cast / Device transfer
// ============================================================================

Array *insight_jl_cast(const Array *x, int32_t dtype) {
  return new Array(cast(*x, static_cast<DType>(dtype)));
}

Array *insight_jl_to_device(const Array *x, int32_t device_type) {
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(x->to(place));
}

// ============================================================================
// Indexing
// ============================================================================

Array *insight_jl_take(const Array *x, const Array *indices, int32_t has_axis,
                       int32_t axis) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(take(*x, *indices, ax));
}

Array *insight_jl_nonzero(const Array *x) { return new Array(nonzero(*x)); }

Array *insight_jl_sort(const Array *x, int32_t axis, int32_t descending) {
  return new Array(sort(*x, axis, descending != 0));
}

// ============================================================================
// Manipulation
// ============================================================================

Array *insight_jl_concat(const Array **arrays, int32_t count, int32_t axis) {
  std::vector<Array> vec;
  for (int32_t i = 0; i < count; i++) {
    vec.push_back(*arrays[i]);
  }
  return new Array(concat(vec, axis));
}

Array *insight_jl_reshape(const Array *x, const int64_t *dims, int32_t ndim) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  return new Array(x->reshape(shape));
}

Array *insight_jl_transpose(const Array *x) {
  return new Array(x->transpose());
}

Array *insight_jl_copy(const Array *x) { return new Array(x->copy()); }

// ============================================================================
// Additional Unary (Phase D)
// ============================================================================

JL_UNARY_OP(exp2, exp2)
JL_UNARY_OP(expm1, expm1)
JL_UNARY_OP(log1p, log1p)
JL_UNARY_OP(cbrt, cbrt)
JL_UNARY_OP(reciprocal, reciprocal)
JL_UNARY_OP(asinh, asinh)
JL_UNARY_OP(acosh, acosh)
JL_UNARY_OP(atanh, atanh)
JL_UNARY_OP(trunc_unary, trunc)
JL_UNARY_OP(deg2rad, deg2rad)
JL_UNARY_OP(rad2deg, rad2deg)

// ============================================================================
// Complex (Phase D)
// ============================================================================

int32_t insight_jl_is_complex(const Array *x) { return is_complex(*x) ? 1 : 0; }
int32_t insight_jl_has_complex_shape(const Array *x) {
  return has_complex_shape(*x) ? 1 : 0;
}
Array *insight_jl_to_complex(const Array *real) {
  return new Array(to_complex(*real));
}
Array *insight_jl_to_complex2(const Array *real, const Array *imag) {
  return new Array(to_complex(*real, *imag));
}
Array *insight_jl_as_complex(const Array *x) {
  return new Array(as_complex(*x));
}
Array *insight_jl_as_real(const Array *x) { return new Array(as_real(*x)); }
Array *insight_jl_real_part(const Array *z) { return new Array(real(*z)); }
Array *insight_jl_imag_part(const Array *z) { return new Array(imag(*z)); }

// ============================================================================
// Additional Reduction (Phase D)
// ============================================================================

Array *insight_jl_cummax(const Array *x, int32_t axis) {
  return new Array(cummax(*x, axis));
}
Array *insight_jl_cummin(const Array *x, int32_t axis) {
  return new Array(cummin(*x, axis));
}
Array *insight_jl_sem(const Array *x, int32_t has_axis, int32_t axis,
                      int32_t keepdims, int32_t ddof) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(sem(*x, ax, keepdims != 0, ddof));
}
Array *insight_jl_count_nonzero(const Array *x, int32_t has_axis, int32_t axis,
                                int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(count_nonzero(*x, ax, keepdims != 0));
}
Array *insight_jl_median(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(median(*x, ax, keepdims != 0));
}
Array *insight_jl_quantile(const Array *x, double q, int32_t has_axis,
                           int32_t axis, int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(quantile(*x, q, ax, keepdims != 0));
}
Array *insight_jl_quantile_arr(const Array *x, const Array *q, int32_t has_axis,
                               int32_t axis, int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(quantile(*x, *q, ax, keepdims != 0));
}
Array *insight_jl_percentile(const Array *x, double q, int32_t has_axis,
                             int32_t axis, int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(percentile(*x, q, ax, keepdims != 0));
}
Array *insight_jl_nansum(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nansum(*x, ax, keepdims != 0));
}
Array *insight_jl_nanmean(const Array *x, int32_t has_axis, int32_t axis,
                          int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nanmean(*x, ax, keepdims != 0));
}
Array *insight_jl_nanmax(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nanmax(*x, ax, keepdims != 0));
}
Array *insight_jl_nanmin(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nanmin(*x, ax, keepdims != 0));
}
Array *insight_jl_nanstd(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims, int32_t ddof) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nanstd(*x, ax, keepdims != 0, ddof));
}
Array *insight_jl_nanvar(const Array *x, int32_t has_axis, int32_t axis,
                         int32_t keepdims, int32_t ddof) {
  std::optional<int> ax = has_axis ? std::optional<int>(axis) : std::nullopt;
  return new Array(nanvar(*x, ax, keepdims != 0, ddof));
}

// ============================================================================
// Additional Manipulation (Phase D)
// ============================================================================

Array *insight_jl_permute(const Array *x, const int32_t *axes, int32_t naxes) {
  std::vector<int> ax(axes, axes + naxes);
  return new Array(permute(*x, ax));
}
Array *insight_jl_swapaxes(const Array *x, int32_t axis1, int32_t axis2) {
  return new Array(swapaxes(*x, axis1, axis2));
}
Array *insight_jl_moveaxis(const Array *x, int32_t source,
                           int32_t destination) {
  return new Array(moveaxis(*x, source, destination));
}
Array *insight_jl_fliplr(const Array *x) { return new Array(fliplr(*x)); }
Array *insight_jl_flipud(const Array *x) { return new Array(flipud(*x)); }
Array *insight_jl_rot90(const Array *x, int32_t k, const int32_t *axes,
                        int32_t naxes) {
  std::vector<int> ax(axes, axes + naxes);
  return new Array(rot90(*x, k, ax));
}
Array *insight_jl_diag(const Array *x, int32_t k) {
  return new Array(diag(*x, k));
}
Array *insight_jl_diagonal(const Array *x, int32_t offset, int32_t axis1,
                           int32_t axis2) {
  return new Array(diagonal(*x, offset, axis1, axis2));
}
Array *insight_jl_tril(const Array *x, int32_t k) {
  return new Array(tril(*x, k));
}
Array *insight_jl_triu(const Array *x, int32_t k) {
  return new Array(triu(*x, k));
}
Array *insight_jl_diff(const Array *x, int32_t n, int32_t axis) {
  return new Array(diff(*x, n, axis));
}

// ============================================================================
// Additional Indexing (Phase D)
// ============================================================================

// unique returns 4 arrays. Caller passes 4 output pointers.
// Unused outputs are left as NULL.
void insight_jl_unique(const Array *x, int32_t return_indices,
                       int32_t return_inverse, int32_t return_counts,
                       Array **u_out, Array **idx_out, Array **inv_out,
                       Array **cnt_out) {
  auto res =
      unique(*x, return_indices != 0, return_inverse != 0, return_counts != 0);
  *u_out = new Array(std::move(res.unique));
  *idx_out = new Array(std::move(res.indices));
  *inv_out = new Array(std::move(res.inverse));
  *cnt_out = new Array(std::move(res.counts));
}

// topk returns (values, indices)
void insight_jl_topk(const Array *x, int64_t k, int32_t axis, int32_t largest,
                     int32_t sorted, Array **vals_out, Array **idx_out) {
  auto [vals, idx] = topk(*x, k, axis, largest != 0, sorted != 0);
  *vals_out = new Array(std::move(vals));
  *idx_out = new Array(std::move(idx));
}

Array *insight_jl_gather(const Array *x, int32_t dim, const Array *index) {
  return new Array(gather(*x, dim, *index));
}
Array *insight_jl_scatter(const Array *x, int32_t dim, const Array *index,
                          const Array *src) {
  return new Array(scatter(*x, dim, *index, *src));
}
Array *insight_jl_scatter_add(const Array *x, int32_t dim, const Array *index,
                              const Array *src) {
  return new Array(scatter_add(*x, dim, *index, *src));
}
Array *insight_jl_scatter_reduce(const Array *x, int32_t dim,
                                 const Array *index, const Array *src,
                                 const char *reduce) {
  return new Array(scatter_reduce(*x, dim, *index, *src, std::string(reduce)));
}
Array *insight_jl_interp(const Array *x, const Array *xp, const Array *fp,
                         int32_t has_left, double left, int32_t has_right,
                         double right) {
  std::optional<double> l =
      has_left ? std::optional<double>(left) : std::nullopt;
  std::optional<double> r =
      has_right ? std::optional<double>(right) : std::nullopt;
  return new Array(interp(*x, *xp, *fp, l, r));
}
Array *insight_jl_indices(const int64_t *dims, int32_t ndim, int32_t sparse) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  return new Array(indices(shape, sparse != 0));
}
void insight_jl_ix_(const Array **arrays, int32_t count, Array **out,
                    int32_t *out_count) {
  std::vector<Array> vec;
  for (int32_t i = 0; i < count; i++) {
    vec.push_back(*arrays[i]);
  }
  auto result = ix_(vec);
  *out_count = static_cast<int32_t>(result.size());
  for (size_t i = 0; i < result.size(); i++) {
    out[i] = new Array(std::move(result[i]));
  }
}

// ============================================================================
// Additional Random (Phase D)
// ============================================================================

void insight_jl_seed(uint64_t s) { seed(s); }
uint64_t insight_jl_get_seed() { return get_seed(); }
Array *insight_jl_rand_like(const Array *x) { return new Array(rand_like(*x)); }
Array *insight_jl_randn_like(const Array *x) {
  return new Array(randn_like(*x));
}
Array *insight_jl_exponential(double scale, const int64_t *dims, int32_t ndim,
                              int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(exponential(scale, shape, static_cast<DType>(dtype), place));
}
Array *insight_jl_gamma(double shape_param, double rate, const int64_t *dims,
                        int32_t ndim, int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(
      gamma(shape_param, rate, shape, static_cast<DType>(dtype), place));
}
Array *insight_jl_beta_dist(double a, double b, const int64_t *dims,
                            int32_t ndim, int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(beta(a, b, shape, static_cast<DType>(dtype), place));
}
Array *insight_jl_binomial(int64_t n, double p, const int64_t *dims,
                           int32_t ndim, int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(binomial(n, p, shape, static_cast<DType>(dtype), place));
}
Array *insight_jl_poisson(double lam, const int64_t *dims, int32_t ndim,
                          int32_t dtype, int32_t device_type) {
  Shape shape(std::vector<int64_t>(dims, dims + ndim));
  Place place = device_type == 1 ? GPUPlace(0) : CPUPlace();
  return new Array(poisson(lam, shape, static_cast<DType>(dtype), place));
}

// ============================================================================
// Additional FFT (Phase D)
// ============================================================================

Array *insight_jl_fftshift(const Array *x, int32_t axis) {
  return new Array(fft::fftshift(*x, axis));
}
Array *insight_jl_ifftshift(const Array *x, int32_t axis) {
  return new Array(fft::ifftshift(*x, axis));
}
int32_t insight_jl_next_fast_len(int32_t target) {
  return fft::next_fast_len(target);
}
Array *insight_jl_hfft(const Array *x, int32_t has_n, int64_t n) {
  return new Array(fft::hfft(*x, has_n ? n : -1));
}
Array *insight_jl_ihfft(const Array *x, int32_t has_n, int64_t n) {
  return new Array(fft::ihfft(*x, has_n ? n : -1));
}
Array *insight_jl_rfft2(const Array *x, const int64_t *s, int32_t s_len,
                        const int32_t *axes, int32_t axes_len) {
  std::vector<int64_t> sv(s, s + s_len);
  std::vector<int> av(axes, axes + axes_len);
  return new Array(fft::rfft2(*x, sv, av));
}
Array *insight_jl_irfft2(const Array *x, const int64_t *s, int32_t s_len,
                         const int32_t *axes, int32_t axes_len) {
  std::vector<int64_t> sv(s, s + s_len);
  std::vector<int> av(axes, axes + axes_len);
  return new Array(fft::irfft2(*x, sv, av));
}
Array *insight_jl_rfftn(const Array *x, const int64_t *s, int32_t s_len,
                        const int32_t *axes, int32_t axes_len) {
  std::vector<int64_t> sv(s, s + s_len);
  std::vector<int> av(axes, axes + axes_len);
  return new Array(fft::rfftn(*x, sv, av));
}
Array *insight_jl_irfftn(const Array *x, const int64_t *s, int32_t s_len,
                         const int32_t *axes, int32_t axes_len) {
  std::vector<int64_t> sv(s, s + s_len);
  std::vector<int> av(axes, axes + axes_len);
  return new Array(fft::irfftn(*x, sv, av));
}

// ============================================================================
// Additional Linalg (Phase D)
// ============================================================================

Array *insight_jl_lstsq(const Array *a, const Array *b) {
  return new Array(lstsq(*a, *b));
}
Array *insight_jl_cond(const Array *x, double p) {
  return new Array(cond(*x, p));
}
Array *insight_jl_matrix_rank(const Array *x) {
  return new Array(matrix_rank(*x));
}

} // extern "C"
