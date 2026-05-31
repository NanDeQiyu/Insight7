// backends/cuda/device/cuda_device.cpp
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "insight/c_api/device_ext.h"
#include "insight/c_api/dtype.h"

#include "../registry/cuda_registry.h"

// ========================================================================
// Thread-local error storage
// ========================================================================

static thread_local std::string gpu_last_error_str;

extern "C" void gpu_set_last_error(const char *msg) {
  gpu_last_error_str = msg ? msg : "";
}

extern "C" const char *gpu_get_last_error(void) {
  if (!gpu_last_error_str.empty()) {
    return gpu_last_error_str.c_str();
  }
  // Fall back to CUDA runtime error
  const char *cuda_err = cudaGetErrorString(cudaGetLastError());
  return cuda_err ? cuda_err : "CUDA backend: no error";
}

// ========================================================================
// Device Lifecycle (Optional)
// ========================================================================

static C_Status cuda_initialize(void) {
  cudaError_t err = cudaFree(nullptr); // Force CUDA context creation
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_finalize(void) {
  cudaError_t err = cudaDeviceReset();
  if (err != cudaSuccess && err != cudaErrorNoDevice) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_init_device(C_Device device) {
  if (!device)
    return C_FAILED;
  cudaError_t err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaFree(nullptr); // Force context init on this device
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_deinit_device(C_Device device) {
  return C_SUCCESS; // cudaDeviceReset() in finalize handles this
}

// ========================================================================
// Device Management (Required)
// ========================================================================

static C_Status cuda_set_device(C_Device device) {
  if (!device)
    return C_FAILED;
  cudaError_t err = cudaSetDevice(device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_get_device(C_Device device) {
  if (!device)
    return C_FAILED;
  int id;
  cudaError_t err = cudaGetDevice(&id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  device->id = id;
  return C_SUCCESS;
}

static C_Status cuda_synchronize_device(C_Device device) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_get_device_count(size_t *count) {
  if (!count)
    return C_FAILED;
  int c;
  cudaError_t err = cudaGetDeviceCount(&c);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *count = static_cast<size_t>(c);
  return C_SUCCESS;
}

static C_Status cuda_get_device_list(size_t *devices) {
  if (!devices)
    return C_FAILED;
  size_t count;
  C_Status s = cuda_get_device_count(&count);
  if (s != C_SUCCESS)
    return s;
  for (size_t i = 0; i < count; ++i) {
    devices[i] = i;
  }
  return C_SUCCESS;
}

// ========================================================================
// Memory Management (Required)
// ========================================================================

static C_Status cuda_device_memory_allocate(C_Device device, void **ptr,
                                            size_t size) {
  if (!ptr)
    return C_FAILED;
  if (size == 0) {
    *ptr = nullptr;
    return C_SUCCESS;
  }
  cudaError_t err = cudaMalloc(ptr, size);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_device_memory_deallocate(C_Device device, void *ptr,
                                              size_t size) {
  if (!ptr)
    return C_SUCCESS;
  cudaError_t err = cudaFree(ptr);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_host_memory_allocate(C_Device device, void **ptr,
                                          size_t size) {
  if (!ptr)
    return C_FAILED;
  if (size == 0) {
    *ptr = nullptr;
    return C_SUCCESS;
  }
  cudaError_t err = cudaMallocHost(ptr, size);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_host_memory_deallocate(C_Device device, void *ptr,
                                            size_t size) {
  if (!ptr)
    return C_SUCCESS;
  cudaError_t err = cudaFreeHost(ptr);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_memory_copy_h2d(C_Device device, void *dst,
                                     const void *src, size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_memory_copy_d2h(C_Device device, void *dst,
                                     const void *src, size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_memory_copy_d2d(C_Device device, void *dst,
                                     const void *src, size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_memory_copy_p2p(C_Device dst_device, C_Device src_device,
                                     void *dst, const void *src, size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err =
      cudaMemcpyPeer(dst, dst_device->id, src, src_device->id, size);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_device_memory_set(C_Device device, void *ptr,
                                       unsigned char value, size_t size) {
  if (!ptr || size == 0)
    return C_SUCCESS;
  cudaError_t err = cudaMemset(ptr, value, size);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_device_memory_stats(C_Device device, size_t *total_memory,
                                         size_t *free_memory) {
  if (!total_memory || !free_memory)
    return C_FAILED;
  cudaError_t err = cudaMemGetInfo(free_memory, total_memory);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Asynchronous Memory Operations (Optional)
// ========================================================================

static C_Status cuda_async_memory_copy_h2d(C_Device device, C_Stream stream,
                                           void *dst, const void *src,
                                           size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_async_memory_copy_d2h(C_Device device, C_Stream stream,
                                           void *dst, const void *src,
                                           size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_async_memory_copy_d2d(C_Device device, C_Stream stream,
                                           void *dst, const void *src,
                                           size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_async_memory_copy_p2p(C_Device dst_device,
                                           C_Device src_device, C_Stream stream,
                                           void *dst, const void *src,
                                           size_t size) {
  if (!dst || !src)
    return C_FAILED;
  cudaError_t err =
      cudaMemcpyPeerAsync(dst, dst_device->id, src, src_device->id, size,
                          reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Stream Management (Optional)
// ========================================================================

static C_Status cuda_create_stream(C_Device device, C_Stream *stream) {
  if (!stream)
    return C_FAILED;
  cudaStream_t s;
  cudaError_t err = cudaStreamCreate(&s);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *stream = reinterpret_cast<C_Stream>(s);
  return C_SUCCESS;
}

static C_Status cuda_destroy_stream(C_Device device, C_Stream stream) {
  cudaError_t err = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_query_stream(C_Device device, C_Stream stream) {
  cudaError_t err = cudaStreamQuery(reinterpret_cast<cudaStream_t>(stream));
  if (err == cudaSuccess)
    return C_SUCCESS;
  if (err == cudaErrorNotReady)
    return C_WARNING;
  gpu_set_last_error(cudaGetErrorString(err));
  return C_FAILED;
}

static C_Status cuda_synchronize_stream(C_Device device, C_Stream stream) {
  cudaError_t err =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_stream_add_callback(C_Device device, C_Stream stream,
                                         void (*callback)(C_Device, C_Stream,
                                                          void *, C_Status *),
                                         void *user_data) {
  // Wrap C API callback for CUDA callback
  cudaError_t err = cudaLaunchHostFunc(
      reinterpret_cast<cudaStream_t>(stream),
      [](void *data) {
        // User callback
      },
      user_data);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_stream_wait_event(C_Device device, C_Stream stream,
                                       C_Event event) {
  cudaError_t err =
      cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(stream),
                          reinterpret_cast<cudaEvent_t>(event), 0);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Event Management (Required)
// ========================================================================

static C_Status cuda_create_event(C_Device device, C_Event *event) {
  if (!event)
    return C_FAILED;
  cudaEvent_t e;
  cudaError_t err = cudaEventCreate(&e);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *event = reinterpret_cast<C_Event>(e);
  return C_SUCCESS;
}

static C_Status cuda_destroy_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_record_event(C_Device device, C_Stream stream,
                                  C_Event event) {
  cudaError_t err = cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                                    reinterpret_cast<cudaStream_t>(stream));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_query_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (err == cudaSuccess)
    return C_SUCCESS;
  if (err == cudaErrorNotReady)
    return C_WARNING;
  gpu_set_last_error(cudaGetErrorString(err));
  return C_FAILED;
}

static C_Status cuda_synchronize_event(C_Device device, C_Event event) {
  cudaError_t err = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(event));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

static C_Status cuda_elapsed_time(C_Event start, C_Event end, float *ms) {
  if (!ms)
    return C_FAILED;
  cudaError_t err =
      cudaEventElapsedTime(ms, reinterpret_cast<cudaEvent_t>(start),
                           reinterpret_cast<cudaEvent_t>(end));
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  return C_SUCCESS;
}

// ========================================================================
// Device Information (Required)
// ========================================================================

static C_Status cuda_get_compute_capability(C_Device device,
                                            size_t *capability) {
  if (!capability || !device)
    return C_FAILED;
  int major, minor;
  cudaError_t err = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *capability = static_cast<size_t>(major * 10 + minor);
  return C_SUCCESS;
}

static C_Status cuda_get_runtime_version(C_Device device, size_t *version) {
  if (!version)
    return C_FAILED;
  int v;
  cudaError_t err = cudaRuntimeGetVersion(&v);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *version = static_cast<size_t>(v);
  return C_SUCCESS;
}

static C_Status cuda_get_driver_version(C_Device device, size_t *version) {
  if (!version)
    return C_FAILED;
  int v;
  cudaError_t err = cudaDriverGetVersion(&v);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *version = static_cast<size_t>(v);
  return C_SUCCESS;
}

static C_Status cuda_get_multi_process(C_Device device, size_t *multi_process) {
  if (!multi_process || !device)
    return C_FAILED;
  int mp;
  cudaError_t err =
      cudaDeviceGetAttribute(&mp, cudaDevAttrMultiProcessorCount, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *multi_process = static_cast<size_t>(mp);
  return C_SUCCESS;
}

static C_Status cuda_get_max_threads_per_mp(C_Device device, size_t *threads) {
  if (!threads || !device)
    return C_FAILED;
  int t;
  cudaError_t err = cudaDeviceGetAttribute(
      &t, cudaDevAttrMaxThreadsPerMultiProcessor, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *threads = static_cast<size_t>(t);
  return C_SUCCESS;
}

static C_Status cuda_get_max_threads_per_block(C_Device device,
                                               size_t *threads) {
  if (!threads || !device)
    return C_FAILED;
  int t;
  cudaError_t err =
      cudaDeviceGetAttribute(&t, cudaDevAttrMaxThreadsPerBlock, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  *threads = static_cast<size_t>(t);
  return C_SUCCESS;
}

static C_Status cuda_get_max_grid_dim_size(C_Device device, size_t dims[3]) {
  if (!dims || !device)
    return C_FAILED;
  int x, y, z;
  cudaError_t err;
  err = cudaDeviceGetAttribute(&x, cudaDevAttrMaxGridDimX, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&y, cudaDevAttrMaxGridDimY, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  err = cudaDeviceGetAttribute(&z, cudaDevAttrMaxGridDimZ, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  dims[0] = static_cast<size_t>(x);
  dims[1] = static_cast<size_t>(y);
  dims[2] = static_cast<size_t>(z);
  return C_SUCCESS;
}

static C_Status cuda_get_device_name(C_Device device, char *buf,
                                     size_t buf_size) {
  if (!buf || buf_size == 0 || !device)
    return C_FAILED;
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device->id);
  if (err != cudaSuccess) {
    gpu_set_last_error(cudaGetErrorString(err));
    return C_FAILED;
  }
  std::strncpy(buf, prop.name, buf_size - 1);
  buf[buf_size - 1] = '\0';
  return C_SUCCESS;
}

// ========================================================================
// Profiler (Optional)
// ========================================================================

static C_Status cuda_profiler_create(C_Profiler *prof, const char *name,
                                     int device_id) {
  if (prof)
    *prof = nullptr; // CUDA profiler uses nvtx/cupti externally
  return C_SUCCESS;
}

static C_Status cuda_profiler_destroy(C_Profiler prof) { return C_SUCCESS; }

static C_Status cuda_profiler_start(C_Profiler prof) { return C_SUCCESS; }

static C_Status cuda_profiler_stop(C_Profiler prof) { return C_SUCCESS; }

static C_Status cuda_profiler_reset(C_Profiler prof) { return C_SUCCESS; }

static C_Status cuda_profiler_begin_event(C_Profiler prof, const char *name) {
  return C_SUCCESS; // Use nvtx for detailed profiling
}

static C_Status cuda_profiler_end_event(C_Profiler prof) { return C_SUCCESS; }

// ========================================================================
// InitPluginGPU - GPU Backend Entry Point
// ========================================================================

extern "C" {
INSIGHT_GPU_API C_Status InitPluginGPU(CustomRuntimeParams *params) {
  if (!params || !params->interface) {
    return C_FAILED;
  }

  INSIGHT_CHECK_CUSTOM_DEVICE_VERSION(params);

  C_DeviceInterface *iface = params->interface;
  iface->size = sizeof(C_DeviceInterface);

  // Device Lifecycle
  iface->initialize = cuda_initialize;
  iface->finalize = cuda_finalize;
  iface->init_device = cuda_init_device;
  iface->deinit_device = cuda_deinit_device;

  // Device Management
  iface->set_device = cuda_set_device;
  iface->get_device = cuda_get_device;
  iface->synchronize_device = cuda_synchronize_device;
  iface->get_device_count = cuda_get_device_count;
  iface->get_device_list = cuda_get_device_list;

  // Memory Management
  iface->device_memory_allocate = cuda_device_memory_allocate;
  iface->device_memory_deallocate = cuda_device_memory_deallocate;
  iface->host_memory_allocate = cuda_host_memory_allocate;
  iface->host_memory_deallocate = cuda_host_memory_deallocate;
  iface->memory_copy_h2d = cuda_memory_copy_h2d;
  iface->memory_copy_d2h = cuda_memory_copy_d2h;
  iface->memory_copy_d2d = cuda_memory_copy_d2d;
  iface->memory_copy_p2p = cuda_memory_copy_p2p;
  iface->device_memory_set = cuda_device_memory_set;
  iface->device_memory_stats = cuda_device_memory_stats;

  // Async Memory
  iface->async_memory_copy_h2d = cuda_async_memory_copy_h2d;
  iface->async_memory_copy_d2h = cuda_async_memory_copy_d2h;
  iface->async_memory_copy_d2d = cuda_async_memory_copy_d2d;
  iface->async_memory_copy_p2p = cuda_async_memory_copy_p2p;

  // Streams
  iface->create_stream = cuda_create_stream;
  iface->destroy_stream = cuda_destroy_stream;
  iface->query_stream = cuda_query_stream;
  iface->synchronize_stream = cuda_synchronize_stream;
  iface->stream_add_callback = cuda_stream_add_callback;
  iface->stream_wait_event = cuda_stream_wait_event;

  // Events
  iface->create_event = cuda_create_event;
  iface->destroy_event = cuda_destroy_event;
  iface->record_event = cuda_record_event;
  iface->query_event = cuda_query_event;
  iface->synchronize_event = cuda_synchronize_event;
  iface->elapsed_time = cuda_elapsed_time;

  // Device Info
  iface->get_compute_capability = cuda_get_compute_capability;
  iface->get_runtime_version = cuda_get_runtime_version;
  iface->get_driver_version = cuda_get_driver_version;
  iface->get_multi_process = cuda_get_multi_process;
  iface->get_max_threads_per_mp = cuda_get_max_threads_per_mp;
  iface->get_max_threads_per_block = cuda_get_max_threads_per_block;
  iface->get_max_grid_dim_size = cuda_get_max_grid_dim_size;
  iface->get_device_name = cuda_get_device_name;

  // Profiler
  iface->profiler_create = cuda_profiler_create;
  iface->profiler_destroy = cuda_profiler_destroy;
  iface->profiler_start = cuda_profiler_start;
  iface->profiler_stop = cuda_profiler_stop;
  iface->profiler_reset = cuda_profiler_reset;
  iface->profiler_begin_event = cuda_profiler_begin_event;
  iface->profiler_end_event = cuda_profiler_end_event;

  // Error
  iface->get_last_error = gpu_get_last_error;

  // Reserved
  for (int i = 0; i < 8; ++i) {
    iface->reserved[i] = nullptr;
  }

  // Device identity
  params->device_type = const_cast<char *>("cuda");
  params->sub_device_type = const_cast<char *>("v1.0");

  if (params->register_kernel) {
    gpu_sync_kernels(params->register_kernel);
  }

  return C_SUCCESS;
}
}