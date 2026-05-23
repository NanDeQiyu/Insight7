// backends/cpu/kernels/creation/logspace.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <cmath>

/**
 * @brief CPU kernel for logspace operation.
 *
 * Generates evenly spaced numbers on a logarithmic scale.
 * Values are computed as: base^(start + i * step) where step = (stop - start) / (n - 1).
 * Special case: when n == 1, returns a single element equal to base^start.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double* stop, [3] = double* base
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
extern "C" {

    C_Status logspace_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = static_cast<InsightArray*>(outputs[0]);

        if (!out) {
            cpu_set_last_error("logspace: output array is null");
            return C_FAILED;
        }
        if (!inputs[1] || !inputs[2] || !inputs[3]) {
            cpu_set_last_error("logspace: start, stop, or base is null");
            return C_FAILED;
        }

        double start = *static_cast<double*>(inputs[1]);
        double stop = *static_cast<double*>(inputs[2]);
        double base = *static_cast<double*>(inputs[3]);
        int64_t n = out->numel;

        // Handle the single-element case
        if (n == 1) {
            double val = std::pow(base, start);
            switch (out->dtype) {
            case INSIGHT_DTYPE_F32: {
                float* data = static_cast<float*>(out->data);
                data[0] = static_cast<float>(val);
                break;
            }
            case INSIGHT_DTYPE_F64: {
                double* data = static_cast<double*>(out->data);
                data[0] = val;
                break;
            }
            default: {
                cpu_set_last_error("logspace: only float32/64 supported");
                return C_FAILED;
            }
            }
            return C_SUCCESS;
        }

        // Compute step size for exponent
        double step = (stop - start) / static_cast<double>(n - 1);

        switch (out->dtype) {
        case INSIGHT_DTYPE_F32: {
            float* data = static_cast<float*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<float>(std::pow(base, start + i * step));
            }
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* data = static_cast<double*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = std::pow(base, start + i * step);
            }
            break;
        }
        default: {
            cpu_set_last_error("logspace: only float32/64 supported");
            return C_FAILED;
        }
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register for float types only
REGISTER_CPU_KERNEL(logspace, INSIGHT_DTYPE_F32, logspace_kernel_cpu);
REGISTER_CPU_KERNEL(logspace, INSIGHT_DTYPE_F64, logspace_kernel_cpu);