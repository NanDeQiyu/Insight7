// backends/cpu/device/cpu_device.cpp
#include "insight/c_api/device_ext.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>   // must come before psapi.h
#include <psapi.h>     // depends on types from windows.h
// clang-format on
#undef interface
#pragma comment(lib, "psapi.lib")
#else
#include <fstream>
#include <sstream>
#include <unistd.h>
#endif

#include "../registry/cpu_registry.h"
#include <unordered_map>
#include <vector>

#ifdef _MSC_VER
#include <stdlib.h>
#define strdup _strdup
#endif

static thread_local std::string cpu_last_error_str;

// CPU event implementation using std::chrono
struct CpuEvent {
  std::chrono::high_resolution_clock::time_point timestamp;
};

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
#ifdef _WIN32
  // Windows: use GlobalMemoryStatusEx for system-wide stats
  MEMORYSTATUSEX mem_stat;
  mem_stat.dwLength = sizeof(mem_stat);
  if (!GlobalMemoryStatusEx(&mem_stat)) {
    if (total_memory)
      *total_memory = 0;
    if (free_memory)
      *free_memory = 0;
    return C_FAILED;
  }
  if (total_memory)
    *total_memory = static_cast<size_t>(mem_stat.ullTotalPhys);
  if (free_memory)
    *free_memory = static_cast<size_t>(mem_stat.ullAvailPhys);
#else
  // Linux: read /proc/meminfo for MemTotal and MemAvailable
  std::ifstream proc_mem("/proc/meminfo");
  std::string line;
  size_t mem_total = 0;
  size_t mem_free = 0;
  if (proc_mem.is_open()) {
    while (std::getline(proc_mem, line)) {
      if (line.find("MemTotal:") == 0) {
        std::istringstream iss(line.substr(9));
        iss >> mem_total;  // in kB
        mem_total *= 1024; // convert to bytes
      } else if (line.find("MemAvailable:") == 0) {
        std::istringstream iss(line.substr(13));
        iss >> mem_free;  // in kB
        mem_free *= 1024; // convert to bytes
      }
      if (mem_total > 0 && mem_free > 0)
        break;
    }
  }

  if (total_memory)
    *total_memory = mem_total;
  if (free_memory)
    *free_memory = mem_free;
#endif
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
// Event Management (Required - CPU events use std::chrono)
// ========================================================================

static C_Status cpu_create_event(C_Device device, C_Event *event) {
  if (!event)
    return C_FAILED;
  auto *e = new (std::nothrow) CpuEvent();
  if (!e)
    return C_FAILED;
  *event = reinterpret_cast<C_Event>(e);
  return C_SUCCESS;
}

static C_Status cpu_destroy_event(C_Device device, C_Event event) {
  if (!event)
    return C_FAILED;
  delete reinterpret_cast<CpuEvent *>(event);
  return C_SUCCESS;
}

static C_Status cpu_record_event(C_Device device, C_Stream stream,
                                 C_Event event) {
  if (!event)
    return C_FAILED;
  auto *e = reinterpret_cast<CpuEvent *>(event);
  e->timestamp = std::chrono::high_resolution_clock::now();
  return C_SUCCESS;
}

static C_Status cpu_query_event(C_Device device, C_Event event) {
  // CPU events are always completed immediately after record
  return C_SUCCESS;
}

static C_Status cpu_synchronize_event(C_Device device, C_Event event) {
  // CPU events need no synchronization (synchronous by nature)
  return C_SUCCESS;
}

static C_Status cpu_elapsed_time(C_Event start, C_Event end, float *ms) {
  if (!start || !end || !ms)
    return C_FAILED;
  auto *s = reinterpret_cast<CpuEvent *>(start);
  auto *e = reinterpret_cast<CpuEvent *>(end);
  auto duration =
      std::chrono::duration<float, std::milli>(e->timestamp - s->timestamp);
  *ms = duration.count();
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
// Profiler (Optional - CPU profiler using std::chrono)
// ========================================================================

struct CpuProfilerEvent {
  std::string name;
  size_t calls = 0;
  double total_ms = 0.0;
  double min_ms = 1e18;
  double max_ms = 0.0;
};

struct CpuProfiler {
  std::string name;
  int device_id;
  std::chrono::high_resolution_clock::time_point event_start;
  std::string current_event_name;
  bool in_event = false;
  bool running = false;
  std::unordered_map<std::string, CpuProfilerEvent> events;
  std::vector<C_ProfilerEvent> report_cache;
};

static C_Status cpu_profiler_create(C_Profiler *prof, const char *name,
                                    int device_id) {
  if (!prof)
    return C_FAILED;
  auto *p = new (std::nothrow) CpuProfiler();
  if (!p)
    return C_FAILED;
  if (name)
    p->name = name;
  p->device_id = device_id;
  *prof = reinterpret_cast<C_Profiler>(p);
  return C_SUCCESS;
}

static C_Status cpu_profiler_destroy(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  delete reinterpret_cast<CpuProfiler *>(prof);
  return C_SUCCESS;
}

static C_Status cpu_profiler_start(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<CpuProfiler *>(prof)->running = true;
  return C_SUCCESS;
}

static C_Status cpu_profiler_stop(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  reinterpret_cast<CpuProfiler *>(prof)->running = false;
  return C_SUCCESS;
}

static C_Status cpu_profiler_reset(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<CpuProfiler *>(prof);
  p->events.clear();
  p->report_cache.clear();
  p->in_event = false;
  return C_SUCCESS;
}

static C_Status cpu_profiler_begin_event(C_Profiler prof, const char *name) {
  if (!prof || !name)
    return C_FAILED;
  auto *p = reinterpret_cast<CpuProfiler *>(prof);
  if (!p->running)
    return C_SUCCESS;
  p->current_event_name = name;
  p->event_start = std::chrono::high_resolution_clock::now();
  p->in_event = true;
  return C_SUCCESS;
}

static C_Status cpu_profiler_end_event(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *p = reinterpret_cast<CpuProfiler *>(prof);
  if (!p->running || !p->in_event)
    return C_SUCCESS;
  auto end = std::chrono::high_resolution_clock::now();
  double ms =
      std::chrono::duration<double, std::milli>(end - p->event_start).count();
  p->in_event = false;

  auto &ev = p->events[p->current_event_name];
  ev.name = p->current_event_name;
  ev.calls++;
  ev.total_ms += ms;
  if (ms < ev.min_ms)
    ev.min_ms = ms;
  if (ms > ev.max_ms)
    ev.max_ms = ms;
  return C_SUCCESS;
}

static C_Status cpu_profiler_get_events(C_Profiler prof,
                                        C_ProfilerEvent **events,
                                        size_t *count) {
  if (!prof || !events || !count)
    return C_FAILED;
  auto *p = reinterpret_cast<CpuProfiler *>(prof);

  // Invalidate old cache
  for (auto &e : p->report_cache) {
    if (e.name) {
      free(const_cast<char *>(e.name));
      e.name = nullptr;
    }
  }
  p->report_cache.clear();

  // Build report
  for (auto &kv : p->events) {
    auto &src = kv.second;
    C_ProfilerEvent ev;
    ev.name = strdup(src.name.c_str());
    ev.calls = src.calls;
    ev.total_ms = static_cast<float>(src.total_ms);
    ev.min_ms = static_cast<float>(src.min_ms < 1e17 ? src.min_ms : 0.0f);
    ev.max_ms = static_cast<float>(src.max_ms);
    p->report_cache.push_back(ev);
  }

  *events = p->report_cache.data();
  *count = p->report_cache.size();
  return C_SUCCESS;
}

// ========================================================================
// Error Handling
// ========================================================================

void cpu_set_last_error(const char *msg) {
  cpu_last_error_str = msg ? msg : "";
}

const char *cpu_get_last_error(void) { return cpu_last_error_str.c_str(); }

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
  iface->profiler_get_events = cpu_profiler_get_events;

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