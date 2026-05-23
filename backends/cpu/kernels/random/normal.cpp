// backends/cpu/kernels/random/normal.cpp
#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

    C_Status normal_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = (InsightArray*)outputs[0];

        if (!out) {
            cpu_set_last_error("normal: output array is null");
            return C_FAILED;
        }

        double mean = *(double*)inputs[1];
        double std = *(double*)inputs[2];

        switch (out->dtype) {
        case INSIGHT_DTYPE_F32:
            RANDOM_FILL_LOOP(float, std::normal_distribution<float>, (mean, std));
            break;
        case INSIGHT_DTYPE_F64:
            RANDOM_FILL_LOOP(double, std::normal_distribution<double>, (mean, std));
            break;
        default:
            cpu_set_last_error("normal: unsupported dtype");
            return C_FAILED;
        }

        return C_SUCCESS;
    }

#ifdef __cplusplus
}
#endif

REGISTER_CPU_KERNEL(normal, INSIGHT_DTYPE_F32, normal_kernel_cpu);
REGISTER_CPU_KERNEL(normal, INSIGHT_DTYPE_F64, normal_kernel_cpu);