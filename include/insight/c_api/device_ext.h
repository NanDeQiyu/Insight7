/**
 * @file device_ext.h
 * @brief Insight Custom Device Interface (C ABI)
 *
 * This header defines the C API for implementing custom device plugins.
 * Any hardware backend (CUDA, ROCm, Ascend, etc.) can be integrated as a plugin
 * by implementing the functions defined in this header.
 *
 * ABI Stability: This header uses pure C interfaces to ensure binary
 * compatibility across different compilers and compiler versions.
 *
 * To implement a plugin:
 * 1. Implement all required functions in C_DeviceInterface
 * 2. Define and export InitPlugin() function
 * 3. Compile as a shared library (.dll/.so)
 *
 * @note This is a C header, not C++. Use extern "C" when including in C++.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include "insight/c_api/dtype.h"
#include "insight/c_api/kernel.h"

#if defined(_WIN32) || defined(_WIN64)
#ifdef INSIGHT_CPU_BACKEND_EXPORTS
#define INSIGHT_CPU_API __declspec(dllexport)
#else
#define INSIGHT_CPU_API
#endif

#ifdef INSIGHT_GPU_BACKEND_EXPORTS
#define INSIGHT_GPU_API __declspec(dllexport)
#else
#define INSIGHT_GPU_API
#endif
#else
#ifdef INSIGHT_CPU_BACKEND_EXPORTS
#define INSIGHT_CPU_API __attribute__((visibility("default")))
#else
#define INSIGHT_CPU_API
#endif

#ifdef INSIGHT_GPU_BACKEND_EXPORTS
#define INSIGHT_GPU_API __attribute__((visibility("default")))
#else
#define INSIGHT_GPU_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*==============================================================================
 * Version Information
 *============================================================================*/

/** @defgroup insight_device_version Version Information
 *  @{
 */

/** Major version of the custom device interface */
#define INSIGHT_CUSTOM_DEVICE_MAJOR_VERSION 1
/** Minor version of the custom device interface */
#define INSIGHT_CUSTOM_DEVICE_MINOR_VERSION 0
/** Patch version of the custom device interface */
#define INSIGHT_CUSTOM_DEVICE_PATCH_VERSION 0

/** @} */

/*==============================================================================
 * Opaque Handles
 *============================================================================*/

/** @defgroup insight_device_handles Opaque Handles
 *  @{
 */

/**
 * @brief Opaque handle to a device.
 *
 * The actual implementation is plugin-defined. The framework only knows
 * that this handle exists.
 */
typedef struct C_Device_st* C_Device;

/**
 * @brief Opaque handle to a stream.
 *
 * Streams are used for asynchronous task execution. Tasks enqueued in the same
 * stream execute in order.
 */
typedef struct C_Stream_st* C_Stream;

/**
 * @brief Opaque handle to an event.
 *
 * Events are used for synchronization between streams and for measuring
 * elapsed time.
 */
typedef struct C_Event_st* C_Event;

/**
 * @brief Opaque handle to a profiler.
 *
 * The profiler is used for performance data collection. The actual structure
 * definition is plugin-specific.
 */
typedef struct C_Profiler_st* C_Profiler;

/**
 * @brief Device descriptor structure.
 *
 * Contains the physical device ID. The framework passes this to plugin functions
 * to identify which device to operate on.
 */
struct C_Device_st {
    int id;                  /**< Physical device ID */
};

/** @} */

/*==============================================================================
 * Device Interface Function Table
 *============================================================================*/

/** @defgroup insight_device_interface Device Interface
 *  @{
 */

/**
 * @brief Complete device interface function table.
 *
 * Plugin must implement all required functions and fill this structure
 * during InitPlugin().
 *
 * Optional functions can be set to NULL. The framework will skip calling them.
 */
typedef struct C_DeviceInterface {
    /**
     * @brief Size of this structure for version checking.
     *
     * Must be set to sizeof(C_DeviceInterface).
     */
    size_t size;

    /*==========================================================================
     * Device Lifecycle (Optional)
     *========================================================================*/

    /**
     * @brief Initialize the hardware backend.
     *
     * Called once when the plugin is first loaded, before any device
     * initialization.
     *
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*initialize)(void);

    /**
     * @brief Finalize the hardware backend.
     *
     * Called once when the framework is shutting down, after all devices
     * have been deinitialized.
     *
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*finalize)(void);

    /**
     * @brief Initialize a specific device.
     *
     * Called for each available device after initialize().
     *
     * @param device Device to initialize (framework fills logical id)
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*init_device)(C_Device device);

    /**
     * @brief Deinitialize a specific device.
     *
     * Called for each device during shutdown, before finalize().
     *
     * @param device Device to deinitialize
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*deinit_device)(C_Device device);

    /*==========================================================================
     * Device Management (Required)
     *========================================================================*/

    /**
     * @brief Set the current device for the calling thread.
     *
     * Subsequent operations (memory allocation, kernel launch) will use
     * this device by default.
     *
     * @param device Device to set as current
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*set_device)(C_Device device);

    /**
     * @brief Get the current device for the calling thread.
     *
     * @param device Output parameter to receive the current device
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_device)(C_Device device);

    /**
     * @brief Synchronize the specified device.
     *
     * Waits for all tasks on the device to complete.
     *
     * @param device Device to synchronize
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*synchronize_device)(C_Device device);

    /**
     * @brief Get the number of available devices.
     *
     * @param count Output parameter to receive device count
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_device_count)(size_t* count);

    /**
     * @brief Get the list of available device IDs.
     *
     * @param devices Output array to receive device IDs (caller allocated,
     *                size = device_count)
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_device_list)(size_t* devices);

    /*==========================================================================
     * Memory Management (Required for basic, Optional for advanced)
     *========================================================================*/

    /**
     * @brief Allocate memory on the device.
     *
     * @param device Device to allocate on
     * @param ptr Output parameter to receive allocated pointer
     * @param size Size in bytes to allocate
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*device_memory_allocate)(C_Device device,
                                       void** ptr,
                                       size_t size);

    /**
     * @brief Deallocate memory on the device.
     *
     * @param device Device where memory was allocated
     * @param ptr Pointer to deallocate
     * @param size Size of the allocation (for compatibility)
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*device_memory_deallocate)(C_Device device,
                                         void* ptr,
                                         size_t size);

    /**
     * @brief Allocate page-locked host memory.
     *
     * Page-locked (pinned) memory accelerates host-to-device and
     * device-to-host transfers.
     *
     * @param device Device to associate with
     * @param ptr Output parameter to receive allocated pointer
     * @param size Size in bytes to allocate
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*host_memory_allocate)(C_Device device,
                                     void** ptr,
                                     size_t size);

    /**
     * @brief Deallocate page-locked host memory.
     *
     * @param device Device associated with the allocation
     * @param ptr Pointer to deallocate
     * @param size Size of the allocation
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*host_memory_deallocate)(C_Device device,
                                       void* ptr,
                                       size_t size);

    /**
     * @brief Copy memory from host to device (synchronous).
     *
     * @param device Destination device
     * @param dst Destination device pointer
     * @param src Source host pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*memory_copy_h2d)(C_Device device,
                                void* dst,
                                const void* src,
                                size_t size);

    /**
     * @brief Copy memory from device to host (synchronous).
     *
     * @param device Source device
     * @param dst Destination host pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*memory_copy_d2h)(C_Device device,
                                void* dst,
                                const void* src,
                                size_t size);

    /**
     * @brief Copy memory from device to device (synchronous).
     *
     * @param device Device where both source and destination reside
     * @param dst Destination device pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*memory_copy_d2d)(C_Device device,
                                void* dst,
                                const void* src,
                                size_t size);

    /**
     * @brief Copy memory from device to device (peer-to-peer, synchronous).
     *
     * @param dst_device Destination device
     * @param src_device Source device
     * @param dst Destination device pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*memory_copy_p2p)(C_Device dst_device,
                                C_Device src_device,
                                void* dst,
                                const void* src,
                                size_t size);

    /**
     * @brief Set device memory to a specified value.
     *
     * @param device Device where memory resides
     * @param ptr Pointer to device memory
     * @param value Value to set (unsigned char)
     * @param size Number of bytes to set
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*device_memory_set)(C_Device device,
                                  void* ptr,
                                  unsigned char value,
                                  size_t size);

    /**
     * @brief Get device memory statistics.
     *
     * @param device Device to query
     * @param total_memory Output parameter for total memory in bytes
     * @param free_memory Output parameter for free memory in bytes
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*device_memory_stats)(C_Device device,
                                    size_t* total_memory,
                                    size_t* free_memory);

    /*==========================================================================
     * Asynchronous Memory Operations (Optional)
     *========================================================================*/

    /**
     * @brief Asynchronous host to device memory copy.
     *
     * If not implemented, the framework will use synchronous version.
     *
     * @param device Destination device
     * @param stream Stream to enqueue the copy operation
     * @param dst Destination device pointer
     * @param src Source host pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*async_memory_copy_h2d)(C_Device device,
                                      C_Stream stream,
                                      void* dst,
                                      const void* src,
                                      size_t size);

    /**
     * @brief Asynchronous device to host memory copy.
     *
     * If not implemented, the framework will use synchronous version.
     *
     * @param device Source device
     * @param stream Stream to enqueue the copy operation
     * @param dst Destination host pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*async_memory_copy_d2h)(C_Device device,
                                      C_Stream stream,
                                      void* dst,
                                      const void* src,
                                      size_t size);

    /**
     * @brief Asynchronous device to device memory copy.
     *
     * If not implemented, the framework will use synchronous version.
     *
     * @param device Device where both source and destination reside
     * @param stream Stream to enqueue the copy operation
     * @param dst Destination device pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*async_memory_copy_d2d)(C_Device device,
                                      C_Stream stream,
                                      void* dst,
                                      const void* src,
                                      size_t size);

    /**
     * @brief Asynchronous peer-to-peer memory copy.
     *
     * If not implemented, the framework will use synchronous version.
     *
     * @param dst_device Destination device
     * @param src_device Source device
     * @param stream Stream to enqueue the copy operation
     * @param dst Destination device pointer
     * @param src Source device pointer
     * @param size Number of bytes to copy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*async_memory_copy_p2p)(C_Device dst_device,
                                      C_Device src_device,
                                      C_Stream stream,
                                      void* dst,
                                      const void* src,
                                      size_t size);

    /*==========================================================================
     * Stream Management (Optional)
     *========================================================================*/

    /**
     * @brief Create a stream on the specified device.
     *
     * @param device Device to create the stream on
     * @param stream Output parameter to receive stream handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*create_stream)(C_Device device, C_Stream* stream);

    /**
     * @brief Destroy a stream.
     *
     * @param device Device where the stream was created
     * @param stream Stream to destroy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*destroy_stream)(C_Device device, C_Stream stream);

    /**
     * @brief Query whether a stream has completed all tasks.
     *
     * If not implemented, the framework will use synchronize_stream.
     *
     * @param device Device where the stream exists
     * @param stream Stream to query
     * @return C_SUCCESS if completed, C_WARNING if not, error code otherwise
     */
    C_Status (*query_stream)(C_Device device, C_Stream stream);

    /**
     * @brief Synchronize a stream.
     *
     * Waits for all tasks enqueued in the stream to complete.
     *
     * @param device Device where the stream exists
     * @param stream Stream to synchronize
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*synchronize_stream)(C_Device device, C_Stream stream);

    /**
     * @brief Add a host callback to a stream.
     *
     * @param device Device where the stream exists
     * @param stream Stream to add the callback to
     * @param callback Callback function to execute
     * @param user_data User data passed to the callback
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*stream_add_callback)(C_Device device,
                                    C_Stream stream,
                                    void (*callback)(C_Device device,
                                                     C_Stream stream,
                                                     void* user_data,
                                                     C_Status* status),
                                    void* user_data);

    /**
     * @brief Make a stream wait for an event.
     *
     * @param device Device where the stream and event exist
     * @param stream Stream that waits
     * @param event Event to wait for
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*stream_wait_event)(C_Device device,
                                  C_Stream stream,
                                  C_Event event);

    /*==========================================================================
     * Event Management (Required)
     *========================================================================*/

    /**
     * @brief Create an event on the specified device.
     *
     * @param device Device to create the event on
     * @param event Output parameter to receive event handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*create_event)(C_Device device, C_Event* event);

    /**
     * @brief Destroy an event.
     *
     * @param device Device where the event was created
     * @param event Event to destroy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*destroy_event)(C_Device device, C_Event event);

    /**
     * @brief Record an event on a stream.
     *
     * Records the event at the current position in the stream. The event
     * will be marked as completed after all preceding tasks in the stream
     * finish.
     *
     * @param device Device where both stream and event exist
     * @param stream Stream to record on
     * @param event Event to record
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*record_event)(C_Device device,
                             C_Stream stream,
                             C_Event event);

    /**
     * @brief Query whether an event has completed.
     *
     * If not implemented, the framework will use synchronize_event.
     *
     * @param device Device where the event exists
     * @param event Event to query
     * @return C_SUCCESS if completed, C_WARNING if not, error code otherwise
     */
    C_Status (*query_event)(C_Device device, C_Event event);

    /**
     * @brief Synchronize an event.
     *
     * Waits for the recorded position in the stream to be reached.
     *
     * @param device Device where the event exists
     * @param event Event to synchronize
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*synchronize_event)(C_Device device, C_Event event);

    /**
     * @brief Calculate elapsed time between two events.
     *
     * Both events must be recorded on the same device and have been
     * synchronized.
     *
     * @param start Starting event
     * @param end Ending event
     * @param ms Output parameter to receive elapsed time in milliseconds
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*elapsed_time)(C_Event start, C_Event end, float* ms);

    /*==========================================================================
     * Device Information (Required + Optional)
     *========================================================================*/

    /**
     * @brief Get the compute capability of a device (required).
     *
     * @param device Device to query
     * @param capability Output parameter to receive compute capability
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_compute_capability)(C_Device device, size_t* capability);

    /**
     * @brief Get the runtime version (required).
     *
     * @param device Device to query
     * @param version Output parameter to receive version
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_runtime_version)(C_Device device, size_t* version);

    /**
     * @brief Get the driver version (required).
     *
     * @param device Device to query
     * @param version Output parameter to receive version
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_driver_version)(C_Device device, size_t* version);

    /**
     * @brief Get the number of multi-processors in a device (optional).
     *
     * @param device Device to query
     * @param multi_process Output parameter to receive multi-processor count
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_multi_process)(C_Device device, size_t* multi_process);

    /**
     * @brief Get the maximum threads per multi-processor (optional).
     *
     * @param device Device to query
     * @param threads Output parameter to receive thread count
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_max_threads_per_mp)(C_Device device, size_t* threads);

    /**
     * @brief Get the maximum threads per block (optional).
     *
     * @param device Device to query
     * @param threads Output parameter to receive thread count
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_max_threads_per_block)(C_Device device, size_t* threads);

    /**
     * @brief Get the maximum grid dimension size (optional).
     *
     * @param device Device to query
     * @param dims Output array of size 3 to receive grid dimensions
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*get_max_grid_dim_size)(C_Device device, size_t dims[3]);

    /*==========================================================================
     * Profiler (Optional)
     *========================================================================*/

    /**
     * @brief Create a profiler instance.
     *
     * @param prof Output parameter to receive profiler handle
     * @param name Profiler name (for identification)
     * @param device_id Device ID this profiler is associated with
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_create)(C_Profiler* prof,
                                const char* name,
                                int device_id);

    /**
     * @brief Destroy a profiler instance.
     *
     * @param prof Profiler handle to destroy
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_destroy)(C_Profiler prof);

    /**
     * @brief Start profiling.
     *
     * Begins recording performance data for the given profiler instance.
     *
     * @param prof Profiler handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_start)(C_Profiler prof);

    /**
     * @brief Stop profiling.
     *
     * Stops recording performance data for the given profiler instance.
     *
     * @param prof Profiler handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_stop)(C_Profiler prof);

    /**
     * @brief Reset profiling data.
     *
     * Clears all recorded performance data for the given profiler instance.
     *
     * @param prof Profiler handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_reset)(C_Profiler prof);

    /**
     * @brief Begin a named event (for nested profiling).
     *
     * @param prof Profiler handle
     * @param name Event name
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_begin_event)(C_Profiler prof, const char* name);

    /**
     * @brief End the current named event.
     *
     * @param prof Profiler handle
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status (*profiler_end_event)(C_Profiler prof);

    /*==========================================================================
     * Error Handling
     *========================================================================*/

    /**
     * @brief Get the last error message.
     *
     * Returns a human-readable error message for the last failed operation.
     * The returned string must remain valid until the next operation on the
     * same device.
     *
     * @return Pointer to error message string (never NULL)
     */
    const char* (*get_last_error)(void);

    /**
     * @brief Reserved fields for future expansion.
     *
     * Must be set to NULL.
     */
    void* reserved[8];
} C_DeviceInterface;

/** @} */

/*==============================================================================
 * Runtime Parameters and Entry Point
 *============================================================================*/

/** @defgroup insight_device_plugin Plugin Entry Point
 *  @{
 */

/**
 * @brief Version information structure.
 *
 * Used for version checking between framework and plugin.
 */
typedef struct CustomRuntimeVersion {
    size_t major;            /**< Major version */
    size_t minor;            /**< Minor version */
    size_t patch;            /**< Patch version */
} CustomRuntimeVersion;

/**
 * @brief Plugin initialization parameters.
 *
 * The framework fills this structure and passes it to InitPlugin().
 * The plugin must check version compatibility and fill the interface table.
 */
typedef struct CustomRuntimeParams {
    /**
     * @brief Size of this structure for version checking.
     *
     * The plugin should verify this matches sizeof(CustomRuntimeParams).
     */
    size_t size;

    /**
     * @brief Device interface table to be filled by the plugin.
     *
     * The plugin must implement the required functions and fill this structure.
     */
    C_DeviceInterface* interface;

    /**
     * @brief Version information to be filled by the plugin.
     *
     * The plugin fills this using INSIGHT_CHECK_CUSTOM_DEVICE_VERSION macro.
     */
    CustomRuntimeVersion version;

    /**
     * @brief Device type name (e.g., "CUDA", "ROCM", "Ascend").
     *
     * This is exposed to users for device selection.
     */
    char* device_type;

    /**
     * @brief Sub-device type (e.g., "V1.0").
     */
    char* sub_device_type;

    /**
     * @brief Core framework's kernel registration callback.
     *
     * Backend plugins use this function pointer to register their kernels
     * into the core framework's global registry. The plugin should call
     * this for every kernel during InitPlugin() after filling the device
     * interface table.
     *
     * Parameters:
     *   op_name     - Operator name (e.g., "sum", "add", "matmul")
     *   device_type - INSIGHT_DEVICE_CPU or INSIGHT_DEVICE_GPU
     *   dtype       - Data type (InsightDType enum value)
     *   kernel      - Kernel function pointer (InsightKernel signature)
     *
     * @return C_SUCCESS on success, error code otherwise
     */
    C_Status(*register_kernel)(const char* op_name, int32_t device_type, int32_t dtype, InsightKernel kernel);


    /**
     * @brief Reserved fields for future expansion.
     */
    char reserved[32];
} CustomRuntimeParams;

/**
 * @brief Version checking macro.
 *
 * This macro must be called at the beginning of InitPlugin().
 * It checks structure size compatibility and fills the version information.
 *
 * @param params The CustomRuntimeParams pointer passed to InitPlugin()
 *
 * Example usage:
 * @code
 * C_Status InitPlugin(CustomRuntimeParams* params) {
 *     INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params);
 *     // ... fill interface table
 *     return C_SUCCESS;
 * }
 * @endcode
 */
#define INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params) \
    do { \
        if ((params)->size != sizeof(CustomRuntimeParams) || \
            (params)->interface->size != sizeof(C_DeviceInterface)) { \
            return C_ERROR; \
        } \
        (params)->version.major = INSIGHT_CUSTOM_DEVICE_MAJOR_VERSION; \
        (params)->version.minor = INSIGHT_CUSTOM_DEVICE_MINOR_VERSION; \
        (params)->version.patch = INSIGHT_CUSTOM_DEVICE_PATCH_VERSION; \
    } while(0)

 /**
  * @brief CPU backend plugin entry point.
  *
  * Every CPU backend plugin must implement and export this function.
  * The framework calls this function when loading the CPU backend.
  *
  * @param params Plugin initialization parameters
  * @return C_SUCCESS on successful initialization, error code otherwise
  */
INSIGHT_CPU_API C_Status InitPluginCPU(CustomRuntimeParams* params);

/**
 * @brief GPU backend plugin entry point.
 *
 * Every GPU backend plugin (CUDA, ROCm, Ascend, etc.) must implement
 * and export this function. The framework calls this function when
 * loading the GPU backend.
 *
 * @param params Plugin initialization parameters
 * @return C_SUCCESS on successful initialization, error code otherwise
 */
INSIGHT_GPU_API C_Status InitPluginGPU(CustomRuntimeParams* params);

/** @} */

#ifdef __cplusplus
}
#endif