// src/core/profiler.cpp
#include "insight/core/profiler.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include <cstring>

using namespace ins;

// ========================================================================
// Internal timer structure
// ========================================================================

struct InsightTimer_st {
  C_Event start_event = nullptr;
  C_Event stop_event = nullptr;
  C_Device_st device = {};
  DeviceKind kind = DeviceKind::CPU;
  const C_DeviceInterface *iface = nullptr;
  C_Stream default_stream = nullptr; // Only used for GPU
};

// ========================================================================
// C API Implementation
// ========================================================================

extern "C" {

C_Status insight_timer_create(const InsightPlace *place, InsightTimer *timer) {
  if (!place || !timer)
    return C_FAILED;

  auto *t = new (std::nothrow) InsightTimer_st();
  if (!t)
    return C_FAILED;

  t->kind = (place->device_type == INSIGHT_DEVICE_GPU) ? DeviceKind::GPU
                                                       : DeviceKind::CPU;
  t->device.id = place->device_id;

  t->iface = get_device_interface(t->kind);
  if (!t->iface) {
    delete t;
    return C_FAILED;
  }

  // Create start and stop events
  if (t->iface->create_event) {
    if (t->iface->create_event(&t->device, &t->start_event) != C_SUCCESS) {
      delete t;
      return C_FAILED;
    }
    if (t->iface->create_event(&t->device, &t->stop_event) != C_SUCCESS) {
      t->iface->destroy_event(&t->device, t->start_event);
      delete t;
      return C_FAILED;
    }
  }

  *timer = t;
  return C_SUCCESS;
}

C_Status insight_timer_destroy(InsightTimer timer) {
  if (!timer)
    return C_FAILED;

  if (timer->iface) {
    if (timer->start_event)
      timer->iface->destroy_event(&timer->device, timer->start_event);
    if (timer->stop_event)
      timer->iface->destroy_event(&timer->device, timer->stop_event);
  }

  delete timer;
  return C_SUCCESS;
}

C_Status insight_timer_start(InsightTimer timer) {
  if (!timer || !timer->iface || !timer->iface->record_event)
    return C_FAILED;
  return timer->iface->record_event(&timer->device, timer->default_stream,
                                    timer->start_event);
}

C_Status insight_timer_stop(InsightTimer timer) {
  if (!timer || !timer->iface || !timer->iface->record_event)
    return C_FAILED;

  C_Status status = timer->iface->record_event(
      &timer->device, timer->default_stream, timer->stop_event);
  if (status != C_SUCCESS)
    return status;

  // Synchronize to ensure accurate timing
  if (timer->iface->synchronize_event)
    return timer->iface->synchronize_event(&timer->device, timer->stop_event);

  return C_SUCCESS;
}

C_Status insight_timer_elapsed_ms(InsightTimer timer, float *ms) {
  if (!timer || !ms || !timer->iface || !timer->iface->elapsed_time)
    return C_FAILED;
  return timer->iface->elapsed_time(timer->start_event, timer->stop_event, ms);
}

C_Status insight_timer_reset(InsightTimer timer) {
  if (!timer || !timer->iface)
    return C_FAILED;

  // Destroy old events
  if (timer->iface->destroy_event) {
    if (timer->start_event)
      timer->iface->destroy_event(&timer->device, timer->start_event);
    if (timer->stop_event)
      timer->iface->destroy_event(&timer->device, timer->stop_event);
  }

  // Create new events
  timer->start_event = nullptr;
  timer->stop_event = nullptr;

  if (timer->iface->create_event) {
    if (timer->iface->create_event(&timer->device, &timer->start_event) !=
        C_SUCCESS)
      return C_FAILED;
    if (timer->iface->create_event(&timer->device, &timer->stop_event) !=
        C_SUCCESS) {
      timer->iface->destroy_event(&timer->device, timer->start_event);
      timer->start_event = nullptr;
      return C_FAILED;
    }
  }

  return C_SUCCESS;
}

} // extern "C"

// ========================================================================
// Profiler C API Implementation (wrapper stores HAL iface)
// ========================================================================

struct ProfilerWrapper {
  C_Profiler prof;
  const C_DeviceInterface *iface;
};

extern "C" {

C_Status insight_profiler_create(const InsightPlace *place, const char *name,
                                 C_Profiler *prof) {
  if (!place || !prof)
    return C_FAILED;

  DeviceKind kind = (place->device_type == INSIGHT_DEVICE_GPU)
                        ? DeviceKind::GPU
                        : DeviceKind::CPU;
  int device_id = place->device_id;

  const C_DeviceInterface *iface = get_device_interface(kind);
  if (!iface || !iface->profiler_create)
    return C_FAILED;

  auto *wrapper = new (std::nothrow) ProfilerWrapper();
  if (!wrapper)
    return C_FAILED;
  wrapper->iface = iface;
  wrapper->prof = nullptr;

  if (iface->profiler_create(&wrapper->prof, name, device_id) != C_SUCCESS) {
    delete wrapper;
    return C_FAILED;
  }

  *prof = reinterpret_cast<C_Profiler>(wrapper);
  return C_SUCCESS;
}

C_Status insight_profiler_destroy(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (wrapper->iface && wrapper->iface->profiler_destroy && wrapper->prof)
    wrapper->iface->profiler_destroy(wrapper->prof);
  delete wrapper;
  return C_SUCCESS;
}

C_Status insight_profiler_start(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_start || !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_start(wrapper->prof);
}

C_Status insight_profiler_stop(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_stop || !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_stop(wrapper->prof);
}

C_Status insight_profiler_reset(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_reset || !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_reset(wrapper->prof);
}

C_Status insight_profiler_begin_event(C_Profiler prof, const char *name) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_begin_event ||
      !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_begin_event(wrapper->prof, name);
}

C_Status insight_profiler_end_event(C_Profiler prof) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_end_event || !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_end_event(wrapper->prof);
}

C_Status insight_profiler_get_events(C_Profiler prof, C_ProfilerEvent **events,
                                     size_t *count) {
  if (!prof)
    return C_FAILED;
  auto *wrapper = reinterpret_cast<ProfilerWrapper *>(prof);
  if (!wrapper->iface || !wrapper->iface->profiler_get_events || !wrapper->prof)
    return C_FAILED;
  return wrapper->iface->profiler_get_events(wrapper->prof, events, count);
}

} // extern "C"

// ========================================================================
// C++ Profiler class (uses C API, which wraps HAL)
// ========================================================================

namespace ins {

Profiler::Profiler(const Place &place) {
  InsightPlace p;
  p.device_type = (place.kind() == DeviceKind::GPU) ? INSIGHT_DEVICE_GPU
                                                    : INSIGHT_DEVICE_CPU;
  p.device_id = place.device_id();

  if (insight_profiler_create(&p, "profiler", &impl_) != C_SUCCESS) {
    impl_ = nullptr;
  }
}

Profiler::~Profiler() {
  if (impl_)
    insight_profiler_destroy(impl_);
}

void Profiler::start() {
  if (impl_)
    insight_profiler_start(impl_);
}

void Profiler::stop() {
  if (impl_)
    insight_profiler_stop(impl_);
}

void Profiler::reset() {
  if (impl_)
    insight_profiler_reset(impl_);
}

void Profiler::begin_event(const char *name) {
  if (impl_)
    insight_profiler_begin_event(impl_, name);
}

void Profiler::end_event() {
  if (impl_)
    insight_profiler_end_event(impl_);
}

void Profiler::report(FILE *fp) {
  if (!impl_)
    return;

  C_ProfilerEvent *events = nullptr;
  size_t count = 0;
  if (insight_profiler_get_events(impl_, &events, &count) != C_SUCCESS)
    return;

  if (count == 0) {
    fprintf(fp, "  [Profiler] no events recorded\n");
    return;
  }

  size_t max_name_len = 5;
  for (size_t i = 0; i < count; ++i) {
    size_t len = events[i].name ? strlen(events[i].name) : 0;
    if (len > max_name_len)
      max_name_len = len;
  }
  if (max_name_len > 40)
    max_name_len = 40;

  fprintf(fp, "\n  %-*s  %7s  %12s  %10s  %10s\n", (int)max_name_len, "Event",
          "Calls", "Total(ms)", "Avg(ms)", "Max(ms)");
  for (size_t i = 0; i < max_name_len + 55; ++i)
    fprintf(fp, "\u2500");
  fprintf(fp, "\n");

  for (size_t i = 0; i < count; ++i) {
    auto &ev = events[i];
    const char *name = ev.name ? ev.name : "(unnamed)";
    float avg = ev.calls > 0 ? ev.total_ms / ev.calls : 0.0f;
    fprintf(fp, "  %-*s  %7zu  %12.3f  %10.4f  %10.4f\n", (int)max_name_len,
            name, ev.calls, ev.total_ms, avg, ev.max_ms);
  }
  fprintf(fp, "\n");
}

} // namespace ins
