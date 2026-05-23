// include/insight/c_api/place.h
#pragma once
#include <cstdint>
#include "insight/c_api/dtype.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef enum {
        INSIGHT_DEVICE_CPU = 0,
        INSIGHT_DEVICE_GPU = 1
    } InsightDeviceType;

    typedef struct {
        int32_t device_type;  // InsightDeviceType
        int32_t device_id;    // GPU ID，0 if use CPU
    } InsightPlace;

    /**
     * @brief Get all available devices as string list.
     *
     * Returns a NULL-terminated array of device strings.
     * Format: "cpu", "gpu:0", "gpu:1", etc.
     *
     * @param devices_out Output pointer for device string array (caller must free)
     * @return C_SUCCESS on success, C_FAILED on error
     */
    C_Status insight_get_available_devices(char*** devices_out);

    /**
     * @brief Free device string array returned by insight_get_available_devices.
     *
     * @param devices Device string array to free
     */
    void insight_free_device_list(char** devices);

#ifdef __cplusplus
}
#endif