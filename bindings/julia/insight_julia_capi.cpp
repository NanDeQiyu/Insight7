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

} // extern "C"
