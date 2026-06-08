// include/insight/core/profiler.h
#pragma once
#include "insight/c_api/profiler.h"
#include "insight/core/place.h"
#include <cstdio>
#include <string>
#include <vector>

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
  explicit Timer(const Place &place = CPUPlace());
  ~Timer();
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
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
  void start();
  void stop();
  float elapsed_ms() const;
  void reset();
  bool started() const;

private:
  InsightTimer impl_ = nullptr;
  Place place_;
  bool started_ = false;
};

/**
 * @brief Lightweight profiler for operator-level timing.
 *
 * Uses the HAL profiler interface to record begin/end event pairs and
 * accumulate statistics (calls, total/avg/min/max) per named event.
 *
 * Usage:
 * @code
 *   Profiler prof(place);
 *   prof.start();
 *   {
 *     ProfileBlock _(prof, "fft");
 *     auto X = fft::fft(x);
 *   }
 *   {
 *     ProfileBlock _(prof, "mul");
 *     auto Y = mul(X, y);
 *   }
 *   prof.stop();
 *   prof.report();  // prints table to stdout
 * @endcode
 */
class Profiler {
public:
  explicit Profiler(const Place &place = CPUPlace());
  ~Profiler();
  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

  /// Start recording events
  void start();
  /// Stop recording events
  void stop();
  /// Clear all recorded data
  void reset();
  /// Begin a named event
  void begin_event(const char *name);
  /// End the current event
  void end_event();
  /**
   * @brief Print a formatted report to a file.
   *
   * Output:
   *   Event           Calls  Total(ms)   Avg(ms)   Max(ms)
   *   ────────────────────────────────────────────────────
   *   fft                10     89.34      8.93     12.45
   */
  void report(FILE *fp = stdout);

private:
  C_Profiler impl_ = nullptr;
};

/**
 * @brief RAII guard for Profiler::begin_event/end_event.
 *
 * Automatically calls begin_event on construction and end_event on
 * destruction, ensuring correct pairing even with exceptions.
 *
 * Usage:
 * @code
 *   Profiler prof(place);
 *   prof.start();
 *   {
 *     ProfileBlock _(prof, "my_op");
 *     // ... work ...
 *   }  // end_event called automatically
 * @endcode
 */
struct ProfileBlock {
  Profiler &prof;
  ProfileBlock(Profiler &p, const char *name_) : prof(p) {
    prof.begin_event(name_);
  }
  ~ProfileBlock() { prof.end_event(); }
  ProfileBlock(const ProfileBlock &) = delete;
  ProfileBlock &operator=(const ProfileBlock &) = delete;
};

} // namespace ins
