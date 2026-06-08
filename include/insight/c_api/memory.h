// include/insight/c_api/memory.h
#pragma once
#include "insight/c_api/device_ext.h"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get memory statistics for a device.
 *
 * For GPU devices, uses cudaMemGetInfo.
 * For CPU devices, reads process memory info (/proc/self/status on Linux,
 * GetProcessMemoryInfo on Windows).
 *
 * @param device_type Device type (0=CPU, 1=GPU)
 * @param device_id Device index
 * @param total_bytes Output: total allocated/used bytes
 * @param free_bytes Output: free/available bytes
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_device_memory_info(int32_t device_type, int32_t device_id,
                                    size_t *total_bytes, size_t *free_bytes);

#ifdef __cplusplus
}
#endif
