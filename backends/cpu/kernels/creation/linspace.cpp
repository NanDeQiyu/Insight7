// backends/cpu/kernels/creation/linspace.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"

/**
 * @brief CPU kernel for linspace operation.
 *
 * Generates evenly spaced numbers over a specified interval.
 * For n points: values = start + i * (stop - start) / (n - 1) for i = 0 to n-1.
 * Special case: when n == 1, returns a single element equal to start.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double* stop
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
extern "C" {

    C_Status linspace_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = static_cast<InsightArray*>(outputs[0]);

        if (!out) {
            cpu_set_last_error("linspace: output array is null");
            return C_FAILED;
        }
        if (!inputs[1] || !inputs[2]) {
            cpu_set_last_error("linspace: start or stop is null");
            return C_FAILED;
        }

        double start = *static_cast<double*>(inputs[1]);
        double stop = *static_cast<double*>(inputs[2]);
        int64_t n = out->numel;

        // Handle the single-element case
        if (n == 1) {
            switch (out->dtype) {
            case INSIGHT_DTYPE_F32: {
                float* data = static_cast<float*>(out->data);
                data[0] = static_cast<float>(start);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* data = static_cast<double*>(out->data);
                data[0] = start;
                break;
            }
            default: {
                cpu_set_last_error("linspace: only float32/64 supported");
                return C_FAILED;
            }
            }
            return C_SUCCESS;
        }

        // Compute step size
        double step = (stop - start) / static_cast<double>(n - 1);

        switch (out->dtype) {
        case INSIGHT_DTYPE_F32: {
            float* data = static_cast<float*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(start + i * step);
            }
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* data = static_cast<double*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = start + i * step;
            }
            break;
        }
        default: {
            cpu_set_last_error("linspace: only float32/64 supported");
            return C_FAILED;
        }
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register for float types only
REGISTER_CPU_KERNEL(linspace, INSIGHT_DTYPE_F32, linspace_kernel_cpu);
REGISTER_CPU_KERNEL(linspace, INSIGHT_DTYPE_F64, linspace_kernel_cpu);