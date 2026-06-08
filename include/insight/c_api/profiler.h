// include/insight/c_api/profiler.h
#pragma once
#include "insight/c_api/device_ext.h"
#include "insight/c_api/place.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a high-resolution timer.
 *
 * The timer uses the device's native event mechanism:
 * - GPU: cudaEvent (cudaEventCreate/Record/Synchronize/ElapsedTime)
 * - CPU: std::chrono::high_resolution_clock
 *
 * Usage:
 * @code
 *   InsightPlace cpu_place = {INSIGHT_DEVICE_CPU, 0};
 *   InsightTimer timer;
 *   insight_timer_create(&cpu_place, &timer);
 *   insight_timer_start(timer);
 *   // ... work ...
 *   insight_timer_stop(timer);
 *   float ms;
 *   insight_timer_elapsed_ms(timer, &ms);
 *   insight_timer_destroy(timer);
 * @endcode
 */
typedef struct InsightTimer_st *InsightTimer;

/**
 * @brief Create a timer on the specified device.
 *
 * Allocates event resources for the given place. For CPU devices,
 * this is a lightweight std::chrono wrapper. For GPU devices,
 * this creates cudaEvent objects.
 *
 * @param place Target device (CPU or GPU with device_id)
 * @param timer Output parameter for the timer handle
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_create(const InsightPlace *place, InsightTimer *timer);

/**
 * @brief Destroy a timer and release associated event resources.
 *
 * @param timer Timer handle to destroy
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_destroy(InsightTimer timer);

/**
 * @brief Record a start event on the timer's device.
 *
 * For GPU, this records a cudaEvent on the default stream. For CPU,
 * this records the current high_resolution_clock time.
 *
 * @param timer Timer handle
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_start(InsightTimer timer);

/**
 * @brief Record a stop event and synchronize.
 *
 * For GPU, this records a cudaEvent and synchronizes to ensure accurate
 * measurement of the actual kernel execution time. For CPU, this records
 * the current time (no sync needed).
 *
 * @param timer Timer handle
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_stop(InsightTimer timer);

/**
 * @brief Get the elapsed time in milliseconds between start and stop.
 *
 * @param timer Timer handle
 * @param ms Output parameter for elapsed time in milliseconds
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_elapsed_ms(InsightTimer timer, float *ms);

/**
 * @brief Reset the timer (re-arm for reuse).
 *
 * Destroys and recreates the underlying events, allowing the timer
 * to be reused for another measurement cycle.
 *
 * @param timer Timer handle
 * @return C_SUCCESS on success, C_FAILED on error
 */
C_Status insight_timer_reset(InsightTimer timer);

#ifdef __cplusplus
}
#endif
