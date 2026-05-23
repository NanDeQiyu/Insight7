// backends/cpu/kernels/linalg/lu_unpack.cpp
/**
 * @file lu_unpack.cpp
 * @brief CPU kernel to unpack LU decomposition into P, L, U.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void lu_unpack_f32(const float* lu, const int* ipiv,
                          float* p, float* l, float* u, int n) {
    // P is permutation matrix
    memset(p, 0, n * n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        p[i * n + (ipiv[i] - 1)] = 1.0f;
    }
    // L and U
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float val = lu[i * n + j];
            if (i == j) {
                l[i * n + j] = 1.0f;
                u[i * n + j] = val;
            } else if (i > j) {
                l[i * n + j] = val;
                u[i * n + j] = 0.0f;
            } else {
                l[i * n + j] = 0.0f;
                u[i * n + j] = val;
            }
        }
    }
}

static void lu_unpack_f64(const double* lu, const int* ipiv,
                          double* p, double* l, double* u, int n) {
    memset(p, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        p[i * n + (ipiv[i] - 1)] = 1.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = lu[i * n + j];
            if (i == j) {
                l[i * n + j] = 1.0;
                u[i * n + j] = val;
            } else if (i > j) {
                l[i * n + j] = val;
                u[i * n + j] = 0.0;
            } else {
                l[i * n + j] = 0.0;
                u[i * n + j] = val;
            }
        }
    }
}

C_Status lu_unpack_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* P_arr = (InsightArray*)outputs[0];
    InsightArray* L_arr = (InsightArray*)outputs[1];
    InsightArray* U_arr = (InsightArray*)outputs[2];
    InsightArray* LU = (InsightArray*)inputs[3];
    InsightArray* pivots = (InsightArray*)inputs[4];

    if (!P_arr || !L_arr || !U_arr || !LU || !pivots) {
        cpu_set_last_error("lu_unpack: null array pointer");
        return C_FAILED;
    }
    if (!cpu_ensure_contiguous(LU) || !cpu_ensure_contiguous(pivots) ||
        !cpu_ensure_contiguous(P_arr) || !cpu_ensure_contiguous(L_arr) ||
        !cpu_ensure_contiguous(U_arr))
        return C_FAILED;

    int n = (int)LU->dims[0];
    if (LU->dims[1] != n) {
        cpu_set_last_error("lu_unpack: LU must be square");
        return C_FAILED;
    }

    const int* ipiv = (const int*)pivots->data;
    if (LU->dtype == INSIGHT_DTYPE_F32) {
        lu_unpack_f32((float*)LU->data, ipiv,
                      (float*)P_arr->data, (float*)L_arr->data, (float*)U_arr->data, n);
    } else {
        lu_unpack_f64((double*)LU->data, ipiv,
                      (double*)P_arr->data, (double*)L_arr->data, (double*)U_arr->data, n);
    }
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(lu_unpack, INSIGHT_DTYPE_F32, lu_unpack_kernel_cpu);
REGISTER_CPU_KERNEL(lu_unpack, INSIGHT_DTYPE_F64, lu_unpack_kernel_cpu);
