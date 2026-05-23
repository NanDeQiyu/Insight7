// backends/cpu/kernels/creation/arange.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <string>

/**
 * @brief CPU kernel for arange operation.
 *
 * Generates a 1D array with evenly spaced values within a given interval.
 * Values are computed as: start + i * step for i = 0 to numel-1.
 *
 * @param inputs  [0] = InsightArray* output, [1] = double* start, [2] = double* step
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
extern "C" {

    C_Status arange_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = static_cast<InsightArray*>(outputs[0]);

        if (!out) {
            cpu_set_last_error("arange: output array is null");
            return C_FAILED;
        }
        if (!inputs[1] || !inputs[2]) {
            cpu_set_last_error("arange: start or step is null");
            return C_FAILED;
        }

        double start = *static_cast<double*>(inputs[1]);
        double step = *static_cast<double*>(inputs[2]);
        int64_t n = out->numel;

        switch (out->dtype) {
        case INSIGHT_DTYPE_I32: {
            int32_t* data = static_cast<int32_t*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<int32_t>(start + i * step);
            }
            break;
        }
        case INSIGHT_DTYPE_I64: {
            int64_t* data = static_cast<int64_t*>(out->data);
            for (int64_t i = 0; i < n; ++i) {
                data[i] = static_cast<int64_t>(start + i * step);
            }
            break;
        }
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
            cpu_set_last_error(("arange: unsupported dtype " + std::to_string(out->dtype)).c_str());
            return C_FAILED;
        }
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register for supported types
REGISTER_CPU_KERNEL(arange, INSIGHT_DTYPE_I32, arange_kernel_cpu);
REGISTER_CPU_KERNEL(arange, INSIGHT_DTYPE_I64, arange_kernel_cpu);
REGISTER_CPU_KERNEL(arange, INSIGHT_DTYPE_F32, arange_kernel_cpu);
REGISTER_CPU_KERNEL(arange, INSIGHT_DTYPE_F64, arange_kernel_cpu);