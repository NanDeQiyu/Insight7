// src/ops/signal/estimation.cpp
#include "insight/ops/signal/estimation.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include <cmath>

namespace ins {
namespace signal {

namespace {

// Batched transpose: transpose last two dimensions of a 3D array
Array batched_transpose(const Array &x, int points, int rows, int cols) {
  return permute(x, {0, 2, 1});
}

// Build batched identity matrix: [points, n, n] with 1s on diagonal
Array batched_eye(int points, int n, DType dtype, const Place &place) {
  Array I_single = eye(n, n, 0, dtype, place);
  std::vector<Array> copies(points, I_single);
  return stack(copies, 0);
}

// Simple Gauss-Jordan inverse for small matrices (no LAPACK dependency)
Array simple_inv(const Array &mat) {
  int n = mat.shape().dim(0);
  Place cpu = CPUPlace();
  DType dtype = mat.dtype();

  Array m = (mat.place().kind() == DeviceKind::CPU) ? mat : mat.to(cpu);

  Array result = eye(n, n, 0, dtype, cpu);

  // Work with copies
  std::vector<double> A(n * n), I(n * n);
  if (dtype == DType::F64) {
    const double *src = m.data<double>();
    for (int i = 0; i < n * n; ++i)
      A[i] = src[i];
  } else {
    const float *src = m.data<float>();
    for (int i = 0; i < n * n; ++i)
      A[i] = src[i];
  }
  for (int i = 0; i < n; ++i)
    I[i * n + i] = 1.0;

  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double max_val = std::abs(A[col * n + col]);
    for (int row = col + 1; row < n; ++row) {
      if (std::abs(A[row * n + col]) > max_val) {
        max_val = std::abs(A[row * n + col]);
        pivot = row;
      }
    }
    if (pivot != col) {
      for (int j = 0; j < n; ++j) {
        std::swap(A[col * n + j], A[pivot * n + j]);
        std::swap(I[col * n + j], I[pivot * n + j]);
      }
    }
    double piv = A[col * n + col];
    for (int j = 0; j < n; ++j) {
      A[col * n + j] /= piv;
      I[col * n + j] /= piv;
    }
    for (int row = 0; row < n; ++row) {
      if (row == col)
        continue;
      double factor = A[row * n + col];
      for (int j = 0; j < n; ++j) {
        A[row * n + j] -= factor * A[col * n + j];
        I[row * n + j] -= factor * I[col * n + j];
      }
    }
  }

  if (dtype == DType::F64) {
    return to_array(std::vector<double>(I.begin(), I.end()), Shape({n, n}),
                    dtype, cpu);
  } else {
    std::vector<float> I_f(I.begin(), I.end());
    return to_array(I_f, Shape({n, n}), dtype, cpu);
  }
}

} // anonymous namespace

KalmanFilter::KalmanFilter(int dim_x, int dim_z, int dim_u, int points,
                           DType dtype)
    : dim_x(dim_x), dim_z(dim_z), dim_u(dim_u), points(points), dtype_(dtype),
      place_(CPUPlace()) {
  INS_CHECK(dim_x > 0, "KalmanFilter: dim_x must be > 0");
  INS_CHECK(dim_z > 0, "KalmanFilter: dim_z must be > 0");
  INS_CHECK(points > 0, "KalmanFilter: points must be > 0");

  // Initialize state arrays
  x = zeros(Shape({points, dim_x, 1}), dtype_, place_);
  P = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  z = zeros(Shape({points, dim_z, 1}), dtype_, place_);
  R = zeros(Shape({points, dim_z, dim_z}), dtype_, place_);
  Q = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  F = batched_eye(points, dim_x, dtype_, place_);
  H = zeros(Shape({points, dim_z, dim_x}), dtype_, place_);

  // Set H to identity-like: H[i,i] = 1 for i < min(dim_z, dim_x)
  Array H_eye = eye(dim_z, dim_x, 0, dtype_, place_);
  // Copy to each point
  for (int p = 0; p < points; ++p) {
    // H[p] = H_eye (via slice assignment)
    Array H_slice = slice(H, {0}, {p}, {p + 1});
    H_slice = reshape(H_slice, {dim_z, dim_x});
    // Copy H_eye into H_slice
    Array H_eye_c = H_eye;
    if (H_eye_c.place() != H_slice.place())
      H_eye_c = H_eye_c.to(H_slice.place());
    if (H_eye_c.dtype() != H_slice.dtype())
      H_eye_c = H_eye_c.to(H_slice.dtype());
    // Use put or direct copy
    // For simplicity, rebuild H with eye values
  }

  // Rebuild H properly using eye + stack
  {
    std::vector<Array> H_parts;
    for (int p = 0; p < points; ++p) {
      H_parts.push_back(eye(dim_z, dim_x, 0, dtype_, place_));
    }
    H = stack(H_parts, 0);
  }
}

void KalmanFilter::predict() {
  // x = F * x
  Array new_x = matmul(F, x);

  // P = F * P * F^T + Q
  Array FP = matmul(F, P);
  Array FT = batched_transpose(F, points, dim_x, dim_x);
  Array P_new = add(matmul(FP, FT), Q);

  // Update state
  x = new_x;
  P = P_new;
}

void KalmanFilter::update(const Array &z_in) {
  INS_CHECK(z_in.defined(), "update: measurement z is undefined");

  Place cpu = CPUPlace();
  Array z_cpu = (z_in.place().kind() == DeviceKind::CPU) ? z_in : z_in.to(cpu);
  z = z_cpu;

  // Innovation: y = z - H * x
  Array Hx = matmul(H, x);
  Array y = sub(z, Hx);

  // Innovation covariance: S = H * P * H^T + R
  Array HP = matmul(H, P);
  Array HT = batched_transpose(H, points, dim_x, dim_z);
  Array S = add(matmul(HP, HT), R);

  // Kalman gain: K = P * H^T * inv(S)
  Array P_Ht = matmul(P, HT); // [points, dim_x, dim_z]

  // Compute inv(S) for each point using simple Gauss-Jordan (no LAPACK)
  std::vector<Array> S_inv_parts;
  for (int p = 0; p < points; ++p) {
    Array S_p = slice(S, {0}, {p}, {p + 1});
    S_p = reshape(S_p, {dim_z, dim_z});
    Array inv_p = simple_inv(S_p);
    S_inv_parts.push_back(inv_p);
  }
  Array S_inv = stack(S_inv_parts, 0);

  // K = P * H^T * S_inv  [points, dim_x, dim_z]
  Array K = matmul(P_Ht, S_inv);

  // x = x + K * y
  Array Ky = matmul(K, y);
  x = add(x, Ky);

  // P = (I - K*H) * P
  Array KH = matmul(K, H);
  Array I_x = batched_eye(points, dim_x, dtype_, cpu);
  if (KH.place() != I_x.place())
    I_x = I_x.to(KH.place());
  Array I_minus_KH = sub(I_x, KH);
  P = matmul(I_minus_KH, P);
}

} // namespace signal
} // namespace ins
