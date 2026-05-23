// backends/cpu/kernels/linalg/lu.cpp
/**
 * @file lu.cpp
 * @brief CPU kernel for LU decomposition with partial pivoting.
 */

#include "common.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static void lu_f32(float* a, int* ipiv, int n) {
    for (int i = 0; i < n; ++i) ipiv[i] = i;
    for (int i = 0; i < n; ++i) {
        // pivot
        int pivot = i;
        float max_val = fabsf(a[i * n + i]);
        for (int k = i + 1; k < n; ++k) {
            float val = fabsf(a[k * n + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        if (max_val < 1e-12f) continue;
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                float tmp = a[i * n + j];
                a[i * n + j] = a[pivot * n + j];
                a[pivot * n + j] = tmp;
            }
            int tmp = ipiv[i];
            ipiv[i] = ipiv[pivot];
            ipiv[pivot] = tmp;
        }
        float pivot_val = a[i * n + i];
        for (int k = i + 1; k < n; ++k) {
            a[k * n + i] /= pivot_val;
        }
        for (int k = i + 1; k < n; ++k) {
            float factor = a[k * n + i];
            for (int j = i + 1; j < n; ++j) {
                a[k * n + j] -= factor * a[i * n + j];
            }
        }
    }
}

static void lu_f64(double* a, int* ipiv, int n) {
    for (int i = 0; i < n; ++i) ipiv[i] = i;
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        double max_val = fabs(a[i * n + i]);
        for (int k = i + 1; k < n; ++k) {
            double val = fabs(a[k * n + i]);
            if (val > max_val) {
                max_val = val;
                pivot = k;
            }
        }
        if (max_val < 1e-12) continue;
        if (pivot != i) {
            for (int j = 0; j < n; ++j) {
                double tmp = a[i * n + j];
                a[i * n + j] = a[pivot * n + j];
                a[pivot * n + j] = tmp;
            }
            int tmp = ipiv[i];
            ipiv[i] = ipiv[pivot];
            ipiv[pivot] = tmp;
        }
        double pivot_val = a[i * n + i];
        for (int k = i + 1; k < n; ++k) {
            a[k * n + i] /= pivot_val;
        }
        for (int k = i + 1; k < n; ++k) {
            double factor = a[k * n + i];
            for (int j = i + 1; j < n; ++j) {
                a[k * n + j] -= factor * a[i * n + j];
            }
        }
    }
}

C_Status lu_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* LU_arr = (InsightArray*)outputs[0];
    InsightArray* pivots_arr = (InsightArray*)outputs[1];
    InsightArray* x = (InsightArray*)inputs[2];
    int pivot_flag = *(int*)inputs[3];

    if (!LU_arr || !pivots_arr || !x) {
        cpu_set_last_error("lu: null array pointer");
        return C_FAILED;
    }
    if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(LU_arr) || !cpu_ensure_contiguous(pivots_arr))
        return C_FAILED;

    int n = (int)x->dims[0];
    if (x->dims[1] != n) {
        cpu_set_last_error("lu: matrix must be square");
        return C_FAILED;
    }

    if (x->dtype == INSIGHT_DTYPE_F32) {
        memcpy(LU_arr->data, x->data, n * n * sizeof(float));
        int* ipiv = (int*)pivots_arr->data;
        lu_f32((float*)LU_arr->data, ipiv, n);
        // Convert to 1?based for LAPACK compatibility
        for (int i = 0; i < n; ++i) ipiv[i] += 1;
    } else {
        memcpy(LU_arr->data, x->data, n * n * sizeof(double));
        int* ipiv = (int*)pivots_arr->data;
        lu_f64((double*)LU_arr->data, ipiv, n);
        for (int i = 0; i < n; ++i) ipiv[i] += 1;
    }
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_F32, lu_kernel_cpu);
REGISTER_CPU_KERNEL(lu, INSIGHT_DTYPE_F64, lu_kernel_cpu);
