// backends/cpu/device/cpu_device.cpp
#include "insight/c_api/device_ext.h"
#include <cstdlib>
#include <cstring>
#include <string>

#include "../registry/cpu_registry.h"

static thread_local std::string cpu_last_error_str;

extern "C" {

// ========================================================================
// Device Lifecycle (Optional)
// ========================================================================

static C_Status cpu_initialize(void) { return C_SUCCESS; }

static C_Status cpu_finalize(void) { return C_SUCCESS; }

static C_Status cpu_init_device(C_Device device) {
  if (device)
    device->id = 0;
  return C_SUCCESS;
}

static C_Status cpu_deinit_device(C_Device device) { return C_SUCCESS; }

// ========================================================================
// Device Management (Required)
// ========================================================================

static C_Status cpu_set_device(C_Device device) { return C_SUCCESS; }

static C_Status cpu_get_device(C_Device device) {
  if (device)
    device->id = 0;
  return C_SUCCESS;
}

static C_Status cpu_synchronize_device(C_Device device) { return C_SUCCESS; }

static C_Status cpu_get_device_count(size_t *count) {
  if (count)
    *count = 1;
  return C_SUCCESS;
}

static C_Status cpu_get_device_list(size_t *devices) {
  if (devices)
    *devices = 0;
  return C_SUCCESS;
}

// ========================================================================
// Memory Management (Required)
// ========================================================================

static C_Status cpu_device_memory_allocate(C_Device device, void **ptr,
                                           size_t size) {
  if (!ptr)
    return C_FAILED;
  if (size == 0) {
    *ptr = nullptr;
    return C_SUCCESS;
  }
  *ptr = std::malloc(size);
  if (!*ptr)
    return C_FAILED;
  return C_SUCCESS;
}

static C_Status cpu_device_memory_deallocate(C_Device device, void *ptr,
                                             size_t size) {
  if (ptr)
    std::free(ptr);
  return C_SUCCESS;
}

static C_Status cpu_host_memory_allocate(C_Device device, void **ptr,
                                         size_t size) {
  return cpu_device_memory_allocate(device, ptr, size);
}

static C_Status cpu_host_memory_deallocate(C_Device device, void *ptr,
                                           size_t size) {
  return cpu_device_memory_deallocate(device, ptr, size);
}

static C_Status cpu_memory_copy_h2d(C_Device device, void *dst, const void *src,
                                    size_t size) {
  if (!dst || !src)
    return C_FAILED;
  std::memcpy(dst, src, size);
  return C_SUCCESS;
}

static C_Status cpu_memory_copy_d2h(C_Device device, void *dst, const void *src,
                                    size_t size) {
  return cpu_memory_copy_h2d(device, dst, src, size);
}

static C_Status cpu_memory_copy_d2d(C_Device device, void *dst, const void *src,
                                    size_t size) {
  return cpu_memory_copy_h2d(device, dst, src, size);
}

static C_Status cpu_memory_copy_p2p(C_Device dst_device, C_Device src_device,
                                    void *dst, const void *src, size_t size) {
  return cpu_memory_copy_h2d(dst_device, dst, src, size);
}

static C_Status cpu_device_memory_set(C_Device device, void *ptr,
                                      unsigned char value, size_t size) {
  if (!ptr || size == 0)
    return C_SUCCESS;
  std::memset(ptr, value, size);
  return C_SUCCESS;
}

static C_Status cpu_device_memory_stats(C_Device device, size_t *total_memory,
                                        size_t *free_memory) {
  if (total_memory)
    *total_memory = 0;
  if (free_memory)
    *free_memory = 0;
  return C_SUCCESS;
}

// ========================================================================
// Asynchronous Memory Operations (Optional - fallback to sync)
// ========================================================================

static C_Status cpu_async_memory_copy_h2d(C_Device device, C_Stream stream,
                                          void *dst, const void *src,
                                          size_t size) {
  return cpu_memory_copy_h2d(device, dst, src, size);
}

static C_Status cpu_async_memory_copy_d2h(C_Device device, C_Stream stream,
                                          void *dst, const void *src,
                                          size_t size) {
  return cpu_memory_copy_d2h(device, dst, src, size);
}

static C_Status cpu_async_memory_copy_d2d(C_Device device, C_Stream stream,
                                          void *dst, const void *src,
                                          size_t size) {
  return cpu_memory_copy_d2d(device, dst, src, size);
}

static C_Status cpu_async_memory_copy_p2p(C_Device dst_device,
                                          C_Device src_device, C_Stream stream,
                                          void *dst, const void *src,
                                          size_t size) {
  return cpu_memory_copy_p2p(dst_device, src_device, dst, src, size);
}

// ========================================================================
// Stream Management (Optional - CPU has no streams, all are no-ops)
// ========================================================================

static C_Status cpu_create_stream(C_Device device, C_Stream *stream) {
  if (stream)
    *stream = nullptr;
  return C_SUCCESS;
}

static C_Status cpu_destroy_stream(C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

static C_Status cpu_query_stream(C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

static C_Status cpu_synchronize_stream(C_Device device, C_Stream stream) {
  return C_SUCCESS;
}

static C_Status cpu_stream_add_callback(C_Device device, C_Stream stream,
                                        void (*callback)(C_Device, C_Stream,
                                                         void *, C_Status *),
                                        void *user_data) {
  if (callback) {
    C_Status status = C_SUCCESS;
    callback(device, stream, user_data, &status);
    return status;
  }
  return C_SUCCESS;
}

static C_Status cpu_stream_wait_event(C_Device device, C_Stream stream,
                                      C_Event event) {
  return C_SUCCESS;
}

// ========================================================================
// Event Management (Required - CPU events are no-ops)
// ========================================================================

static C_Status cpu_create_event(C_Device device, C_Event *event) {
  if (event)
    *event = nullptr;
  return C_SUCCESS;
}

static C_Status cpu_destroy_event(C_Device device, C_Event event) {
  return C_SUCCESS;
}

static C_Status cpu_record_event(C_Device device, C_Stream stream,
                                 C_Event event) {
  return C_SUCCESS;
}

static C_Status cpu_query_event(C_Device device, C_Event event) {
  return C_SUCCESS;
}

static C_Status cpu_synchronize_event(C_Device device, C_Event event) {
  return C_SUCCESS;
}

static C_Status cpu_elapsed_time(C_Event start, C_Event end, float *ms) {
  if (ms)
    *ms = 0.0f;
  return C_SUCCESS;
}

// ========================================================================
// Device Information (Required)
// ========================================================================

static C_Status cpu_get_compute_capability(C_Device device,
                                           size_t *capability) {
  if (capability)
    *capability = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_runtime_version(C_Device device, size_t *version) {
  if (version)
    *version = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_driver_version(C_Device device, size_t *version) {
  if (version)
    *version = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_multi_process(C_Device device, size_t *multi_process) {
  if (multi_process)
    *multi_process = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_max_threads_per_mp(C_Device device, size_t *threads) {
  if (threads)
    *threads = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_max_threads_per_block(C_Device device,
                                              size_t *threads) {
  if (threads)
    *threads = 0;
  return C_SUCCESS;
}

static C_Status cpu_get_max_grid_dim_size(C_Device device, size_t dims[3]) {
  if (dims) {
    dims[0] = 0;
    dims[1] = 0;
    dims[2] = 0;
  }
  return C_SUCCESS;
}

// ========================================================================
// Profiler (Optional - CPU profiler is a no-op)
// ========================================================================

static C_Status cpu_profiler_create(C_Profiler *prof, const char *name,
                                    int device_id) {
  if (prof)
    *prof = nullptr;
  return C_SUCCESS;
}

static C_Status cpu_profiler_destroy(C_Profiler prof) { return C_SUCCESS; }

static C_Status cpu_profiler_start(C_Profiler prof) { return C_SUCCESS; }

static C_Status cpu_profiler_stop(C_Profiler prof) { return C_SUCCESS; }

static C_Status cpu_profiler_reset(C_Profiler prof) { return C_SUCCESS; }

static C_Status cpu_profiler_begin_event(C_Profiler prof, const char *name) {
  return C_SUCCESS;
}

static C_Status cpu_profiler_end_event(C_Profiler prof) { return C_SUCCESS; }

// ========================================================================
// Error Handling
// ========================================================================

void cpu_set_last_error(const char *msg) {
  cpu_last_error_str = msg ? msg : "";
}

const char *cpu_get_last_error(void) {
  return cpu_last_error_str.empty() ? "CPU backend: no error"
                                    : cpu_last_error_str.c_str();
}

// ========================================================================
// InitPlugin - Plugin Entry Point
// ========================================================================

INSIGHT_CPU_API C_Status InitPluginCPU(CustomRuntimeParams *params) {
  if (!params || !params->interface) {
    return C_FAILED;
  }

  INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params);

  C_DeviceInterface *iface = params->interface;
  iface->size = sizeof(C_DeviceInterface);

  // Device Lifecycle
  iface->initialize = cpu_initialize;
  iface->finalize = cpu_finalize;
  iface->init_device = cpu_init_device;
  iface->deinit_device = cpu_deinit_device;

  // Device Management
  iface->set_device = cpu_set_device;
  iface->get_device = cpu_get_device;
  iface->synchronize_device = cpu_synchronize_device;
  iface->get_device_count = cpu_get_device_count;
  iface->get_device_list = cpu_get_device_list;

  // Memory Management
  iface->device_memory_allocate = cpu_device_memory_allocate;
  iface->device_memory_deallocate = cpu_device_memory_deallocate;
  iface->host_memory_allocate = cpu_host_memory_allocate;
  iface->host_memory_deallocate = cpu_host_memory_deallocate;
  iface->memory_copy_h2d = cpu_memory_copy_h2d;
  iface->memory_copy_d2h = cpu_memory_copy_d2h;
  iface->memory_copy_d2d = cpu_memory_copy_d2d;
  iface->memory_copy_p2p = cpu_memory_copy_p2p;
  iface->device_memory_set = cpu_device_memory_set;
  iface->device_memory_stats = cpu_device_memory_stats;

  // Async Memory
  iface->async_memory_copy_h2d = cpu_async_memory_copy_h2d;
  iface->async_memory_copy_d2h = cpu_async_memory_copy_d2h;
  iface->async_memory_copy_d2d = cpu_async_memory_copy_d2d;
  iface->async_memory_copy_p2p = cpu_async_memory_copy_p2p;

  // Streams
  iface->create_stream = cpu_create_stream;
  iface->destroy_stream = cpu_destroy_stream;
  iface->query_stream = cpu_query_stream;
  iface->synchronize_stream = cpu_synchronize_stream;
  iface->stream_add_callback = cpu_stream_add_callback;
  iface->stream_wait_event = cpu_stream_wait_event;

  // Events
  iface->create_event = cpu_create_event;
  iface->destroy_event = cpu_destroy_event;
  iface->record_event = cpu_record_event;
  iface->query_event = cpu_query_event;
  iface->synchronize_event = cpu_synchronize_event;
  iface->elapsed_time = cpu_elapsed_time;

  // Device Info
  iface->get_compute_capability = cpu_get_compute_capability;
  iface->get_runtime_version = cpu_get_runtime_version;
  iface->get_driver_version = cpu_get_driver_version;
  iface->get_multi_process = cpu_get_multi_process;
  iface->get_max_threads_per_mp = cpu_get_max_threads_per_mp;
  iface->get_max_threads_per_block = cpu_get_max_threads_per_block;
  iface->get_max_grid_dim_size = cpu_get_max_grid_dim_size;

  // Profiler
  iface->profiler_create = cpu_profiler_create;
  iface->profiler_destroy = cpu_profiler_destroy;
  iface->profiler_start = cpu_profiler_start;
  iface->profiler_stop = cpu_profiler_stop;
  iface->profiler_reset = cpu_profiler_reset;
  iface->profiler_begin_event = cpu_profiler_begin_event;
  iface->profiler_end_event = cpu_profiler_end_event;

  // Error
  iface->get_last_error = cpu_get_last_error;

  // Reserved
  for (int i = 0; i < 8; ++i) {
    iface->reserved[i] = nullptr;
  }

  // Device identity
  params->device_type = const_cast<char *>("cpu");
  params->sub_device_type = const_cast<char *>("v1.0");

  if (params->register_kernel) {
    cpu_sync_kernels(params->register_kernel);
  }

  return C_SUCCESS;
}
}