// src/ops/signal/estimation.cpp
#include "insight/ops/signal/estimation.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/linalg.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include <cmath>

namespace ins {
namespace signal {

KalmanFilter::KalmanFilter(int dim_x, int dim_z, int dim_u, int points,
                           DType dtype)
    : dim_x(dim_x), dim_z(dim_z), dim_u(dim_u), points(points), dtype_(dtype),
      place_(CPUPlace()) {
  INS_CHECK(dim_x > 0, "KalmanFilter: dim_x must be > 0");
  INS_CHECK(dim_z > 0, "KalmanFilter: dim_z must be > 0");
  INS_CHECK(points > 0, "KalmanFilter: points must be > 0");

  // Initialize state
  x = zeros(Shape({points, dim_x, 1}), dtype_, place_);
  P = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  z = zeros(Shape({points, dim_z, 1}), dtype_, place_);
  R = zeros(Shape({points, dim_z, dim_z}), dtype_, place_);
  Q = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  F = zeros(Shape({points, dim_x, dim_x}), dtype_, place_);
  H = zeros(Shape({points, dim_z, dim_x}), dtype_, place_);

  // Set F and H to identity-like defaults
  // F = I (identity state transition)
  // For each point, set diagonal
  if (dtype_ == DType::F64) {
    double *fd = F.data<double>();
    double *hd = H.data<double>();
    for (int p = 0; p < points; ++p) {
      for (int i = 0; i < dim_x; ++i) {
        fd[p * dim_x * dim_x + i * dim_x + i] = 1.0;
      }
      for (int i = 0; i < std::min(dim_z, dim_x); ++i) {
        hd[p * dim_z * dim_x + i * dim_x + i] = 1.0;
      }
    }
  } else {
    float *fd = F.data<float>();
    float *hd = H.data<float>();
    for (int p = 0; p < points; ++p) {
      for (int i = 0; i < dim_x; ++i) {
        fd[p * dim_x * dim_x + i * dim_x + i] = 1.0f;
      }
      for (int i = 0; i < std::min(dim_z, dim_x); ++i) {
        hd[p * dim_z * dim_x + i * dim_x + i] = 1.0f;
      }
    }
  }
}

void KalmanFilter::predict() {
  // x = F * x
  // P = F * P * F^T + Q
  // All operations are batched over points

  // x = F * x  [points, dim_x, dim_x] @ [points, dim_x, 1]
  Array new_x = matmul(F, x);

  // P = F * P * F^T + Q
  // First compute F * P
  Array FP = matmul(F, P);

  // Compute F^T
  // For batched transpose, we need to transpose the last two dims
  Shape ft_shape({points, dim_x, dim_x});
  Array FT = zeros(ft_shape, dtype_, place_);

  if (dtype_ == DType::F64) {
    const double *fp_d = FP.data<double>();
    const double *ft_src = F.data<double>();
    double *ft_d = FT.data<double>();
    double *p_d = P.data<double>();
    double *q_d = Q.data<double>();
    double *nx_d = new_x.data<double>();
    double *x_d = x.data<double>();

    for (int p = 0; p < points; ++p) {
      int base = p * dim_x * dim_x;
      int base_v = p * dim_x;

      // Transpose F
      for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          ft_d[base + i * dim_x + j] = ft_src[base + j * dim_x + i];
        }
      }

      // FP * F^T
      for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          double sum = 0;
          for (int k = 0; k < dim_x; ++k) {
            sum += fp_d[base + i * dim_x + k] * ft_d[base + k * dim_x + j];
          }
          p_d[base + i * dim_x + j] = sum + q_d[base + i * dim_x + j];
        }
      }

      // Update x
      for (int i = 0; i < dim_x; ++i) {
        x_d[base_v + i] = nx_d[base_v + i];
      }
    }
  } else {
    const float *fp_d = FP.data<float>();
    const float *ft_src = F.data<float>();
    float *ft_d = FT.data<float>();
    float *p_d = P.data<float>();
    float *q_d = Q.data<float>();
    float *nx_d = new_x.data<float>();
    float *x_d = x.data<float>();

    for (int p = 0; p < points; ++p) {
      int base = p * dim_x * dim_x;
      int base_v = p * dim_x;

      for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          ft_d[base + i * dim_x + j] = ft_src[base + j * dim_x + i];
        }
      }

      for (int i = 0; i < dim_x; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          float sum = 0;
          for (int k = 0; k < dim_x; ++k) {
            sum += fp_d[base + i * dim_x + k] * ft_d[base + k * dim_x + j];
          }
          p_d[base + i * dim_x + j] = sum + q_d[base + i * dim_x + j];
        }
      }

      for (int i = 0; i < dim_x; ++i) {
        x_d[base_v + i] = nx_d[base_v + i];
      }
    }
  }
}

void KalmanFilter::update(const Array &z_in) {
  INS_CHECK(z_in.defined(), "update: measurement z is undefined");

  Place cpu = CPUPlace();
  Array z_cpu = (z_in.place().kind() == DeviceKind::CPU) ? z_in : z_in.to(cpu);

  // Store measurement
  z = z_cpu;

  // Innovation: y = z - H * x
  Array Hx = matmul(H, x);

  // Innovation covariance: S = H * P * H^T + R
  Array HP = matmul(H, P);

  // H^T
  Array HT = zeros(Shape({points, dim_x, dim_z}), dtype_, cpu);
  if (dtype_ == DType::F64) {
    const double *hd = H.data<double>();
    double *ht_d = HT.data<double>();
    for (int p = 0; p < points; ++p) {
      int h_base = p * dim_z * dim_x;
      int ht_base = p * dim_x * dim_z;
      for (int i = 0; i < dim_z; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          ht_d[ht_base + j * dim_z + i] = hd[h_base + i * dim_x + j];
        }
      }
    }
  } else {
    const float *hd = H.data<float>();
    float *ht_d = HT.data<float>();
    for (int p = 0; p < points; ++p) {
      int h_base = p * dim_z * dim_x;
      int ht_base = p * dim_x * dim_z;
      for (int i = 0; i < dim_z; ++i) {
        for (int j = 0; j < dim_x; ++j) {
          ht_d[ht_base + j * dim_z + i] = hd[h_base + i * dim_x + j];
        }
      }
    }
  }

  Array HPHt = matmul(HP, HT);
  Array S = HPHt; // S = H*P*H^T + R
  // Add R
  if (dtype_ == DType::F64) {
    double *sd = S.data<double>();
    const double *rd = R.data<double>();
    for (int64_t i = 0; i < S.numel(); ++i)
      sd[i] += rd[i];
  } else {
    float *sd = S.data<float>();
    const float *rd = R.data<float>();
    for (int64_t i = 0; i < S.numel(); ++i)
      sd[i] += rd[i];
  }

  // Kalman gain: K = P * H^T * S^-1
  // For each point, compute K = P * H^T * inv(S)
  // Simplified: batch solve S * K^T = H * P
  // K = P * H^T * inv(S)

  // Compute S_inv for each point
  // For small matrices, use explicit inverse
  Array S_inv = zeros(Shape({points, dim_z, dim_z}), dtype_, cpu);
  if (dtype_ == DType::F64) {
    const double *sd = S.data<double>();
    double *si_d = S_inv.data<double>();

    for (int p = 0; p < points; ++p) {
      // Gauss-Jordan elimination for 2x2 or small matrices
      int base = p * dim_z * dim_z;
      double mat[16]; // max dim_z = 4
      double inv[16] = {0};
      for (int i = 0; i < dim_z; ++i) {
        inv[i * dim_z + i] = 1.0;
        for (int j = 0; j < dim_z; ++j) {
          mat[i * dim_z + j] = sd[base + i * dim_z + j];
        }
      }

      for (int col = 0; col < dim_z; ++col) {
        // Find pivot
        int pivot = col;
        double max_val = std::abs(mat[col * dim_z + col]);
        for (int row = col + 1; row < dim_z; ++row) {
          if (std::abs(mat[row * dim_z + col]) > max_val) {
            max_val = std::abs(mat[row * dim_z + col]);
            pivot = row;
          }
        }
        // Swap rows
        if (pivot != col) {
          for (int j = 0; j < dim_z; ++j) {
            std::swap(mat[col * dim_z + j], mat[pivot * dim_z + j]);
            std::swap(inv[col * dim_z + j], inv[pivot * dim_z + j]);
          }
        }
        // Scale pivot row
        double piv_val = mat[col * dim_z + col];
        for (int j = 0; j < dim_z; ++j) {
          mat[col * dim_z + j] /= piv_val;
          inv[col * dim_z + j] /= piv_val;
        }
        // Eliminate
        for (int row = 0; row < dim_z; ++row) {
          if (row == col)
            continue;
          double factor = mat[row * dim_z + col];
          for (int j = 0; j < dim_z; ++j) {
            mat[row * dim_z + j] -= factor * mat[col * dim_z + j];
            inv[row * dim_z + j] -= factor * inv[col * dim_z + j];
          }
        }
      }

      for (int i = 0; i < dim_z * dim_z; ++i) {
        si_d[base + i] = inv[i];
      }
    }
  } else {
    const float *sd = S.data<float>();
    float *si_d = S_inv.data<float>();

    for (int p = 0; p < points; ++p) {
      int base = p * dim_z * dim_z;
      float mat[16];
      float inv[16] = {0};
      for (int i = 0; i < dim_z; ++i) {
        inv[i * dim_z + i] = 1.0f;
        for (int j = 0; j < dim_z; ++j) {
          mat[i * dim_z + j] = sd[base + i * dim_z + j];
        }
      }

      for (int col = 0; col < dim_z; ++col) {
        int pivot = col;
        float max_val = std::abs(mat[col * dim_z + col]);
        for (int row = col + 1; row < dim_z; ++row) {
          if (std::abs(mat[row * dim_z + col]) > max_val) {
            max_val = std::abs(mat[row * dim_z + col]);
            pivot = row;
          }
        }
        if (pivot != col) {
          for (int j = 0; j < dim_z; ++j) {
            std::swap(mat[col * dim_z + j], mat[pivot * dim_z + j]);
            std::swap(inv[col * dim_z + j], inv[pivot * dim_z + j]);
          }
        }
        float piv_val = mat[col * dim_z + col];
        for (int j = 0; j < dim_z; ++j) {
          mat[col * dim_z + j] /= piv_val;
          inv[col * dim_z + j] /= piv_val;
        }
        for (int row = 0; row < dim_z; ++row) {
          if (row == col)
            continue;
          float factor = mat[row * dim_z + col];
          for (int j = 0; j < dim_z; ++j) {
            mat[row * dim_z + j] -= factor * mat[col * dim_z + j];
            inv[row * dim_z + j] -= factor * inv[col * dim_z + j];
          }
        }
      }

      for (int i = 0; i < dim_z * dim_z; ++i) {
        si_d[base + i] = inv[i];
      }
    }
  }

  // K = P * H^T * S_inv  [points, dim_x, dim_z]
  Array P_Ht = matmul(P, HT);    // [points, dim_x, dim_z]
  Array K = matmul(P_Ht, S_inv); // [points, dim_x, dim_z]

  // Innovation: y = z - H*x  [points, dim_z, 1]
  Array y = z_cpu;
  if (dtype_ == DType::F64) {
    double *yd = y.data<double>();
    const double *hxd = Hx.data<double>();
    for (int64_t i = 0; i < y.numel(); ++i)
      yd[i] -= hxd[i];
  } else {
    float *yd = y.data<float>();
    const float *hxd = Hx.data<float>();
    for (int64_t i = 0; i < y.numel(); ++i)
      yd[i] -= hxd[i];
  }

  // x = x + K * y
  Array Ky = matmul(K, y); // [points, dim_x, 1]
  if (dtype_ == DType::F64) {
    double *xd = x.data<double>();
    const double *kyd = Ky.data<double>();
    for (int64_t i = 0; i < x.numel(); ++i)
      xd[i] += kyd[i];
  } else {
    float *xd = x.data<float>();
    const float *kyd = Ky.data<float>();
    for (int64_t i = 0; i < x.numel(); ++i)
      xd[i] += kyd[i];
  }

  // P = (I - K*H) * P
  Array KH = matmul(K, H); // [points, dim_x, dim_x]
  Array I_minus_KH = zeros(Shape({points, dim_x, dim_x}), dtype_, cpu);
  if (dtype_ == DType::F64) {
    double *ikd = I_minus_KH.data<double>();
    const double *khd = KH.data<double>();
    for (int p = 0; p < points; ++p) {
      int base = p * dim_x * dim_x;
      for (int i = 0; i < dim_x; ++i) {
        ikd[base + i * dim_x + i] = 1.0;
      }
      for (int i = 0; i < dim_x * dim_x; ++i) {
        ikd[base + i] -= khd[base + i];
      }
    }
  } else {
    float *ikd = I_minus_KH.data<float>();
    const float *khd = KH.data<float>();
    for (int p = 0; p < points; ++p) {
      int base = p * dim_x * dim_x;
      for (int i = 0; i < dim_x; ++i) {
        ikd[base + i * dim_x + i] = 1.0f;
      }
      for (int i = 0; i < dim_x * dim_x; ++i) {
        ikd[base + i] -= khd[base + i];
      }
    }
  }

  P = matmul(I_minus_KH, P);
}

} // namespace signal
} // namespace ins
