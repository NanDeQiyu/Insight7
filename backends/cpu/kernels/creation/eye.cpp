// backends/cpu/kernels/creation/eye.cpp
#include "../../registry/cpu_registry.h"
#include "insight/c_api/array.h"
#include <complex>
#include <cstring>

/**
 * @brief CPU kernel for eye operation (identity matrix).
 *
 * Creates a 2D array with ones on the diagonal and zeros elsewhere.
 * The diagonal offset k specifies which diagonal to fill:
 *   - k = 0: main diagonal
 *   - k > 0: super-diagonal (above main)
 *   - k < 0: sub-diagonal (below main)
 *
 * @param inputs  [0] = InsightArray* output (2D matrix), [1] = int64_t*
 * diagonal offset
 * @param outputs [0] = InsightArray* result (same as inputs[0])
 * @return C_SUCCESS on success, C_FAILED on error
 */
extern "C" {

C_Status eye_kernel_cpu(void **inputs, void **outputs) {
  InsightArray *out = static_cast<InsightArray *>(outputs[0]);

  if (!out) {
    cpu_set_last_error("eye: output array is null");
    return C_FAILED;
  }
  if (!inputs[1]) {
    cpu_set_last_error("eye: k is null");
    return C_FAILED;
  }

  // Validate dimensions: must be 2D
  if (out->ndim != 2) {
    cpu_set_last_error("eye: output must be 2D");
    return C_FAILED;
  }

  int64_t k = *static_cast<int64_t *>(inputs[1]);
  int64_t n = out->dims[0]; // number of rows
  int64_t m = out->dims[1]; // number of columns
  int64_t total = n * m;

  // Zero out the entire matrix
  std::memset(out->data, 0, total * insight_dtype_size(out->dtype));

  switch (out->dtype) {
  case INSIGHT_DTYPE_BOOL: {
    bool *data = static_cast<bool *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = true;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_U8: {
    uint8_t *data = static_cast<uint8_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I8: {
    int8_t *data = static_cast<int8_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I16: {
    int16_t *data = static_cast<int16_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I32: {
    int32_t *data = static_cast<int32_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_I64: {
    int64_t *data = static_cast<int64_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_U16: {
    uint16_t *data = static_cast<uint16_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_U32: {
    uint32_t *data = static_cast<uint32_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_U64: {
    uint64_t *data = static_cast<uint64_t *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F32: {
    float *data = static_cast<float *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1.0f;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_F64: {
    double *data = static_cast<double *>(out->data);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = 1.0;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_C32: {
    std::complex<float> *data = static_cast<std::complex<float> *>(out->data);
    std::complex<float> one(1.0f, 0.0f);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = one;
      }
    }
    break;
  }
  case INSIGHT_DTYPE_C64: {
    std::complex<double> *data = static_cast<std::complex<double> *>(out->data);
    std::complex<double> one(1.0, 0.0);
    for (int64_t i = 0; i < n; ++i) {
      int64_t j = i + k;
      if (j >= 0 && j < m) {
        data[i * m + j] = one;
      }
    }
    break;
  }
  default: {
    cpu_set_last_error(
        ("eye: unsupported dtype " + std::to_string(out->dtype)).c_str());
    return C_FAILED;
  }
  }

  return C_SUCCESS;
}

} // extern "C"

// Register for all supported types
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_BOOL, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_U8, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_I8, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_I16, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_I32, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_I64, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_U16, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_U32, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_U64, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_F32, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_F64, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_C32, eye_kernel_cpu);
REGISTER_CPU_KERNEL(eye, INSIGHT_DTYPE_C64, eye_kernel_cpu);