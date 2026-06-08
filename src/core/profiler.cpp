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
// C++ Timer class
// ========================================================================

namespace ins {

Timer::Timer(const Place &place) : place_(place), started_(false) {
  InsightPlace p;
  p.device_type = (place.kind() == DeviceKind::GPU) ? INSIGHT_DEVICE_GPU
                                                    : INSIGHT_DEVICE_CPU;
  p.device_id = place.device_id();

  if (insight_timer_create(&p, &impl_) != C_SUCCESS) {
    INS_THROW("Timer: failed to create timer for device ",
              (place.kind() == DeviceKind::GPU ? "GPU" : "CPU"), ":",
              place.device_id());
  }
}

Timer::~Timer() {
  if (impl_)
    insight_timer_destroy(impl_);
}

void Timer::start() {
  if (!impl_)
    return;
  if (insight_timer_start(impl_) == C_SUCCESS)
    started_ = true;
}

void Timer::stop() {
  if (!impl_)
    return;
  if (insight_timer_stop(impl_) == C_SUCCESS)
    started_ = false;
}

float Timer::elapsed_ms() const {
  if (!impl_)
    return 0.0f;
  float ms = 0.0f;
  insight_timer_elapsed_ms(impl_, &ms);
  return ms;
}

void Timer::reset() {
  if (!impl_)
    insight_timer_reset(impl_);
}

bool Timer::started() const { return started_; }

} // namespace ins
