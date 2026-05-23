// backends/cpu/kernels/random/beta.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

    static void beta_fill_float(float* out_data, int64_t n, int64_t ndim,
        const int64_t* dims, const int64_t* out_strides,
        float a, float b) {
        std::gamma_distribution<float> dist_a(a, 1.0f);
        std::gamma_distribution<float> dist_b(b, 1.0f);

#pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
            float x = dist_a(cpu_get_rng());
            float y = dist_b(cpu_get_rng());
            out_data[off_out] = x / (x + y);
        }
    }

    static void beta_fill_double(double* out_data, int64_t n, int64_t ndim,
        const int64_t* dims, const int64_t* out_strides,
        double a, double b) {
        std::gamma_distribution<double> dist_a(a, 1.0);
        std::gamma_distribution<double> dist_b(b, 1.0);

#pragma omp parallel for
        for (int64_t linear = 0; linear < n; ++linear) {
            int64_t off_out = cpu_offset_from_linear(linear, ndim, dims, out_strides);
            double x = dist_a(cpu_get_rng());
            double y = dist_b(cpu_get_rng());
            out_data[off_out] = x / (x + y);
        }
    }

    C_Status beta_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];

        if (!out) {
            cpu_set_last_error("beta: output array is null");
            return C_FAILED;
        }

        double a = *(double*)inputs[1];
        double b = *(double*)inputs[2];

        int64_t ndim = out->ndim;
        int64_t dims[INSIGHT_MAX_NDIM];
        int64_t out_strides[INSIGHT_MAX_NDIM];
        for (int i = 0; i < ndim; ++i) {
            dims[i] = out->dims[i];
            out_strides[i] = out->strides[i];
        }
        int64_t n = out->numel;

        switch (out->dtype) {
        case INSIGHT_DTYPE_F32:
            beta_fill_float((float*)out->data, n, ndim, dims, out_strides,
                (float)a, (float)b);
            break;
        case INSIGHT_DTYPE_F64:
            beta_fill_double((double*)out->data, n, ndim, dims, out_strides, a, b);
            break;
        default:
            cpu_set_last_error("beta: unsupported dtype");
            return C_FAILED;
        }

        return C_SUCCESS;
    }

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(beta, INSIGHT_DTYPE_F32, beta_kernel_cpu);
REGISTER_CPU_KERNEL(beta, INSIGHT_DTYPE_F64, beta_kernel_cpu);