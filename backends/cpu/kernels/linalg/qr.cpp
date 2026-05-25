// backends/cpu/kernels/linalg/qr.cpp
/**
 * @file qr.cpp
 * @brief CPU kernel for QR decomposition using LAPACK.
 */

#include "common.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

template <typename T>
static void qr_impl_manual(const T *src, T *q, T *r, int m, int n, int mode) {
  int k = std::min(m, n);

  // 复制矩阵到列主序（我们会修改它）
  std::vector<T> A(m * n);
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i + j * m] = src[i * n + j];
    }
  }

  std::vector<T> tau(k);

  // ========== QR分解：逐列进行Householder变换 ==========
  for (int j = 0; j < k; ++j) {
    int len = m - j;
    T *col_ptr = A.data() + j * m + j;

    // 计算当前列的范数
    T norm = 0;
    for (int i = 0; i < len; ++i) {
      norm += col_ptr[i] * col_ptr[i];
    }
    norm = std::sqrt(norm);

    if (norm > std::numeric_limits<T>::epsilon()) {
      T &first = col_ptr[0];
      T sign = (first > 0) ? T(1) : T(-1);
      T u1 = first + sign * norm;
      tau[j] = sign * u1 / norm;

      // 修改第一个元素为 -sign * norm
      first = -sign * norm;

      // 归一化Householder向量（v[0] = 1，其他除以u1）
      T inv_u1 = T(1) / u1;
      for (int i = 1; i < len; ++i) {
        col_ptr[i] *= inv_u1;
      }

      // 应用到剩余的列
#pragma omp parallel for
      for (int col = j + 1; col < n; ++col) {
        T *target_col = A.data() + col * m + j;

        // 计算点积 (v^T * x)，其中v[0]=1
        T dot = target_col[0];
        for (int row = 1; row < len; ++row) {
          dot += col_ptr[row] * target_col[row];
        }
        dot *= tau[j];

        // 更新: x = x - dot * v
        target_col[0] -= dot;
        for (int row = 1; row < len; ++row) {
          target_col[row] -= dot * col_ptr[row];
        }
      }
    } else {
      tau[j] = T(0);
    }
  }

  // ========== 提取R矩阵 ==========
  std::vector<T> R_colmajor(k * n, T(0));
#pragma omp parallel for
  for (int i = 0; i < k; ++i) {
    for (int j = i; j < n; ++j) {
      R_colmajor[i + j * k] = A[i + j * m];
    }
  }

  // ========== 构建Q矩阵 ==========
  std::vector<T> Q_colmajor(m * m, T(0));
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    Q_colmajor[i + i * m] = T(1);
  }

  // 从后向前应用Householder变换
  for (int j = k - 1; j >= 0; --j) {
    int len = m - j;
    T *v_ptr = A.data() + j * m + j;
    v_ptr[0] = T(1); // Householder向量，第一个元素强制为1

    // Q = Q * (I - tau * v * v^T)
#pragma omp parallel for
    for (int col = 0; col < m; ++col) {
      T *Q_col = Q_colmajor.data() + col * m + j;

      // 计算点积 (v^T * Q_col)
      T dot = Q_col[0]; // v[0]=1
      for (int row = 1; row < len; ++row) {
        dot += v_ptr[row] * Q_col[row];
      }
      dot *= tau[j];

      // 更新: Q_col = Q_col - dot * v
      Q_col[0] -= dot;
      for (int row = 1; row < len; ++row) {
        Q_col[row] -= dot * v_ptr[row];
      }
    }
  }

  // ========== 转换结果到行主序 ==========
  if (mode == 0) { // complete
#pragma omp parallel
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        q[i * m + j] = Q_colmajor[i + j * m];
      }
    }
  } else { // reduced
#pragma omp parallel
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        q[i * k + j] = Q_colmajor[i + j * m];
      }
    }
  }

  // 转换R到行主序
#pragma omp parallel
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      r[i * n + j] = R_colmajor[i + j * k];
    }
  }
}

// 对外接口
void qr_f64_impl(const double *src, double *q, double *r, int m, int n,
                 int mode) {
  qr_impl_manual(src, q, r, m, n, mode);
}

void qr_f32_impl(const float *src, float *q, float *r, int m, int n, int mode) {
  qr_impl_manual(src, q, r, m, n, mode);
}

// ============================================================================
// C API kernel entry point (only this needs extern "C")
// ============================================================================

extern "C" {

C_Status qr_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *Q = (InsightArray *)outputs[0];
  InsightArray *R = (InsightArray *)outputs[1];
  InsightArray *x = (InsightArray *)inputs[2];
  int mode = *(int *)inputs[3];

  if (!Q || !R || !x) {
    cpu_set_last_error("qr: null array pointer");
    return C_FAILED;
  }

  // Ensure input is contiguous (LAPACK requirement)
  if (!insight_array_is_contiguous(x)) {
    cpu_set_last_error("qr: input must be contiguous");
    return C_FAILED;
  }

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  if (x->dtype == INSIGHT_DTYPE_F64) {
    qr_f64_impl((double *)x->data, (double *)Q->data, (double *)R->data, m, n,
                mode);
  } else if (x->dtype == INSIGHT_DTYPE_F32) {
    qr_f32_impl((float *)x->data, (float *)Q->data, (float *)R->data, m, n,
                mode);
  } else {
    cpu_set_last_error("qr: unsupported dtype (only F32/F64)");
    return C_FAILED;
  }

  return C_SUCCESS;
}

} // extern "C"

// Register kernels
REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_F64, qr_kernel_cpu);
REGISTER_CPU_KERNEL(qr, INSIGHT_DTYPE_F32, qr_kernel_cpu);