// backends/cpu/kernels/linalg/trace.cpp
/**
 * @file trace.cpp
 * @brief CPU kernel for trace of a square matrix.
 */

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

static float trace_f32(const float* data, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += data[i * n + i];
    }
    return sum;
}

static double trace_f64(const double* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i * n + i];
    }
    return sum;
}

C_Status trace_kernel_cpu(void** inputs, void** outputs) {
    InsightArray* out = (InsightArray*)outputs[0];
    InsightArray* x = (InsightArray*)inputs[1];

    if (!out || !x) {
        cpu_set_last_error("trace: null array pointer");
        return C_FAILED;
    }
    if (!cpu_ensure_contiguous(x)) return C_FAILED;

    int n = (int)x->dims[0];
    if (x->dims[1] != n) {
        cpu_set_last_error("trace: matrix must be square");
        return C_FAILED;
    }

    if (x->dtype == INSIGHT_DTYPE_F32) {
        float* dst = (float*)out->data;
        *dst = trace_f32((float*)x->data, n);
    } else {
        double* dst = (double*)out->data;
        *dst = trace_f64((double*)x->data, n);
    }
    return C_SUCCESS;
}

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(trace, INSIGHT_DTYPE_F32, trace_kernel_cpu);
REGISTER_CPU_KERNEL(trace, INSIGHT_DTYPE_F64, trace_kernel_cpu);
