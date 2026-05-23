// backends/cpu/kernels/creation/full.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include "insight/c_api/dtype.h"
#include <cstring>
#include <complex>

extern "C" {

    /**
     * @brief CPU implementation of the full kernel.
     *
     * Fills the output array with a constant scalar value.
     * The fill value is always passed as a double* in inputs[1],
     * and the kernel casts it to the output array's dtype internally.
     *
     * @param inputs  [0] = InsightArray* output, [1] = double* fill_value
     * @param outputs [0] = InsightArray* result (same as inputs[0])
     * @return C_SUCCESS on success, C_FAILED on error
     */
    C_Status full_kernel_cpu(void** inputs, void** outputs) {
        InsightArray* out = static_cast<InsightArray*>(outputs[0]);

        if (!out) {
            cpu_set_last_error("full_kernel_cpu: output array is null");
            return C_FAILED;
        }
        if (!inputs[1]) {
            cpu_set_last_error("full_kernel_cpu: fill_value is null");
            return C_FAILED;
        }

        double fill_val = *static_cast<double*>(inputs[1]);
        int64_t n = out->numel;
        int32_t dtype = out->dtype;

        switch (dtype) {
        case INSIGHT_DTYPE_BOOL: {
            bool* dst = static_cast<bool*>(out->data);
            bool val = (fill_val != 0.0);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_U8: {
            uint8_t* dst = static_cast<uint8_t*>(out->data);
            uint8_t val = static_cast<uint8_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_I8: {
            int8_t* dst = static_cast<int8_t*>(out->data);
            int8_t val = static_cast<int8_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_I16: {
            int16_t* dst = static_cast<int16_t*>(out->data);
            int16_t val = static_cast<int16_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_I32: {
            int32_t* dst = static_cast<int32_t*>(out->data);
            int32_t val = static_cast<int32_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_I64: {
            int64_t* dst = static_cast<int64_t*>(out->data);
            int64_t val = static_cast<int64_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_U16: {
            uint16_t* dst = static_cast<uint16_t*>(out->data);
            uint16_t val = static_cast<uint16_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_U32: {
            uint32_t* dst = static_cast<uint32_t*>(out->data);
            uint32_t val = static_cast<uint32_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_U64: {
            uint64_t* dst = static_cast<uint64_t*>(out->data);
            uint64_t val = static_cast<uint64_t>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_F32: {
            float* dst = static_cast<float*>(out->data);
            float val = static_cast<float>(fill_val);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_F64: {
            double* dst = static_cast<double*>(out->data);
            for (int64_t i = 0; i < n; ++i) dst[i] = fill_val;
            break;
        }
        case INSIGHT_DTYPE_C32: {
            std::complex<float>* dst = static_cast<std::complex<float>*>(out->data);
            std::complex<float> val(static_cast<float>(fill_val), 0.0f);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        case INSIGHT_DTYPE_C64: {
            std::complex<double>* dst = static_cast<std::complex<double>*>(out->data);
            std::complex<double> val(fill_val, 0.0);
            for (int64_t i = 0; i < n; ++i) dst[i] = val;
            break;
        }
        default:
            cpu_set_last_error(("full_kernel_cpu: unsupported dtype " + std::to_string(dtype)).c_str());
            return C_FAILED;
        }

        return C_SUCCESS;
    }

} // extern "C"

// Register for all supported types
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_BOOL, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_U8, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_I8, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_I16, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_I32, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_I64, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_U16, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_U32, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_U64, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_F32, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_F64, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_C32, full_kernel_cpu);
REGISTER_CPU_KERNEL(full, INSIGHT_DTYPE_C64, full_kernel_cpu);