// backends/cpu/kernels/linalg/matrix_power.cpp
/**
 * @file matrix_power.cpp
 * @brief CPU kernel for integer power of a square matrix.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static void mat_mul_f32(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void mat_mul_f64(const double* a, const double* b, double* c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

static void mat_power_f32(const float* src, float* dst, int n, int exp) {
    // dst = I
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    float* base = (float*)malloc(n * n * sizeof(float));
    memcpy(base, src, n * n * sizeof(float));
    float* tmp = (float*)malloc(n * n * sizeof(float));
    while (exp > 0) {
        if (exp & 1) {
            mat_mul_f32(dst, base, tmp, n);
            memcpy(dst, tmp, n * n * sizeof(float));
        }
        mat_mul_f32(base, base, tmp, n);
        memcpy(base, tmp, n * n * sizeof(float));
        exp >>= 1;
    }
    free(base);
    free(tmp);
}

static void mat_power_f64(const double* src, double* dst, int n, int exp) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dst[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    double* base = (double*)malloc(n * n * sizeof(double));
    memcpy(base, src, n * n * sizeof(double));
    double* tmp = (double*)malloc(n * n * sizeof(double));
    while (exp > 0) {
        if (exp & 1) {
            mat_mul_f64(dst, base, tmp, n);
            memcpy(dst, tmp, n * n * sizeof(double));
        }
        mat_mul_f64(base, base, tmp, n);
        memcpy(base, tmp, n * n * sizeof(double));
        exp >>= 1;
    }
    free(base);
    free(tmp);
}

C_Status matrix_power_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* x = (InsightArray*)inputs[1];
    int n = *(int*)inputs[2];

    if (!out || !x) {
        cpu_set_last_error("matrix_power: null array pointer");
        return C_FAILED;
    }
    if (!cpu_ensure_contiguous(x) || !cpu_ensure_contiguous(out))
        return C_FAILED;

    int size = (int)x->dims[0];
    if (x->dims[1] != size) {
        cpu_set_last_error("matrix_power: matrix must be square");
        return C_FAILED;
    }

    if (x->dtype == INSIGHT_DTYPE_F32) {
        mat_power_f32((float*)x->data, (float*)out->data, size, n);
    } else {
        mat_power_f64((double*)x->data, (double*)out->data, size, n);
    }
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(matrix_power, INSIGHT_DTYPE_F32, matrix_power_kernel_cpu);
REGISTER_CPU_KERNEL(matrix_power, INSIGHT_DTYPE_F64, matrix_power_kernel_cpu);
