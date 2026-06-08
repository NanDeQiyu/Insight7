// include/insight/core/profiler.h
#pragma once
#include "insight/c_api/profiler.h"
#include "insight/core/place.h"

namespace ins {

/**
 * @brief High-resolution timer for measuring execution time.
 *
 * RAII wrapper around the C API InsightTimer. Uses the device's native
 * event mechanism to measure elapsed time:
 *   - GPU: cudaEvent (accurate kernel execution time)
 *   - CPU: std::chrono::high_resolution_clock
 *
 * Usage:
 * @code
 *   Timer t(CPUPlace());
 *   t.start();
 *   // ... work ...
 *   t.stop();
 *   std::cout << "Elapsed: " << t.elapsed_ms() << " ms" << std::endl;
 * @endcode
 *
 * Timer is non-copyable.
 */
class Timer {
public:
  /**
   * @brief Create a timer for the specified device.
   *
   * @param place Device to measure on (CPUPlace or GPUPlace)
   */
  explicit Timer(const Place &place = CPUPlace());

  /**
   * @brief Destroy the timer and release event resources.
   */
  ~Timer();

  // Non-copyable
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;

  // Moveable
  Timer(Timer &&other) noexcept
      : impl_(other.impl_), place_(other.place_), started_(other.started_) {
    other.impl_ = nullptr;
  }
  Timer &operator=(Timer &&other) noexcept {
    if (this != &other) {
      if (impl_)
        insight_timer_destroy(impl_);
      impl_ = other.impl_;
      place_ = other.place_;
      started_ = other.started_;
      other.impl_ = nullptr;
    }
    return *this;
  }

  /// Record the start event on the device.
  void start();

  /// Record the stop event and synchronize.
  void stop();

  /// Get elapsed time between start and stop in milliseconds.
  float elapsed_ms() const;

  /// Reset the timer for reuse.
  void reset();

  /// Check if the timer has been started and not yet stopped.
  bool started() const;

private:
  InsightTimer impl_ = nullptr;
  Place place_;
  bool started_ = false;
};

} // namespace ins
