// backends/cpu/kernels/linalg/pinv.cpp
/**
 * @file pinv.cpp
 * @brief CPU kernel for Moore-Penrose pseudo-inverse using SVD.
 */

#include "common.h"

#ifdef INSIGHT_USE_OPENBLAS

extern "C" {
void inv_f32(const float *src, float *dst, int n);
void inv_f64(const double *src, double *dst, int n);
}
template <typename T> static void inv_general(const T *src, T *dst, int n);

// 特化版本
template <> void inv_general<float>(const float *src, float *dst, int n) {
  inv_f32(src, dst, n);
}

template <> void inv_general<double>(const double *src, double *dst, int n) {
  inv_f64(src, dst, n);
}

template <typename T>
static void pinv(const T *src, T *dst, int m, int n, double rcond) {
  if (m == n) {
    // 方阵：直接求逆
    inv_general(src, dst, m); // 复用你已有的 inv 实现
    return;
  }

  if (m > n) {
    // 高矩阵：pinv = (A^T A)^{-1} A^T
    // 1. 计算 A^T A (n x n)
    T *ATA = (T *)malloc(n * n * sizeof(T));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        T sum = 0;
        for (int k = 0; k < m; ++k) {
          sum += src[k * n + i] * src[k * n + j];
        }
        ATA[i * n + j] = sum;
      }
    }

    // 2. 求逆
    T *ATA_inv = (T *)malloc(n * n * sizeof(T));
    inv_general(ATA, ATA_inv, n);

    // 3. 计算 pinv = ATA_inv * A^T (n x m)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        T sum = 0;
        for (int k = 0; k < n; ++k) {
          sum += ATA_inv[i * n + k] * src[j * n + k];
        }
        dst[i * m + j] = sum;
      }
    }

    free(ATA);
    free(ATA_inv);
  } else {
    // 宽矩阵：pinv = A^T (A A^T)^{-1}
    // 1. 计算 A A^T (m x m)
    T *AAT = (T *)malloc(m * m * sizeof(T));
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < m; ++j) {
        T sum = 0;
        for (int k = 0; k < n; ++k) {
          sum += src[i * n + k] * src[j * n + k];
        }
        AAT[i * m + j] = sum;
      }
    }

    // 2. 求逆
    T *AAT_inv = (T *)malloc(m * m * sizeof(T));
    inv_general(AAT, AAT_inv, m);

    // 3. 计算 pinv = A^T * AAT_inv (n x m)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        T sum = 0;
        for (int k = 0; k < m; ++k) {
          sum += src[i + k * n] * AAT_inv[k * m + j];
        }
        dst[i * m + j] = sum;
      }
    }

    free(AAT);
    free(AAT_inv);
  }
}

C_Status pinv_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = (InsightArray *)outputs[0];
  InsightArray *x = (InsightArray *)inputs[1];
  double rcond = *(double *)inputs[2];

  if (!out || !x) {
    cpu_set_last_error("pinv: null array pointer");
    return C_FAILED;
  }
  if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
    return C_FAILED;

  int m = (int)x->dims[0];
  int n = (int)x->dims[1];

  if (x->dtype == INSIGHT_DTYPE_F32) {
    pinv((float *)x->data, (float *)out->data, m, n, rcond);
  } else {
    pinv((double *)x->data, (double *)out->data, m, n, rcond);
  }
  return C_SUCCESS;
}

REGISTER_CPU_KERNEL(pinv, INSIGHT_DTYPE_F32, pinv_kernel_cpu);
REGISTER_CPU_KERNEL(pinv, INSIGHT_DTYPE_F64, pinv_kernel_cpu);

#endif // INSIGHT_USE_OPENBLAS
