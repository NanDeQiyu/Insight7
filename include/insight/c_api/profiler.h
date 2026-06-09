// include/insight/c_api/profiler.h
#pragma once
#include "insight/c_api/device_ext.h"
#include "insight/c_api/place.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*==============================================================================
 * Timer API — single-shot point-to-point timing using HAL C_Event
 *============================================================================*/

typedef struct InsightTimer_st *InsightTimer;

C_Status insight_timer_create(const InsightPlace *place, InsightTimer *timer);
C_Status insight_timer_destroy(InsightTimer timer);
C_Status insight_timer_start(InsightTimer timer);
C_Status insight_timer_stop(InsightTimer timer);
C_Status insight_timer_elapsed_ms(InsightTimer timer, float *ms);
C_Status insight_timer_reset(InsightTimer timer);

/*==============================================================================
 * Profiler API — multi-event aggregated timing using HAL C_Profiler
 *============================================================================*/

/**
 * @brief Create a profiler instance.
 * @param place Device kind and ID
 * @param name Optional profiler name (may be NULL)
 * @param prof Output: opaque profiler handle
 */
C_Status insight_profiler_create(const InsightPlace *place, const char *name,
                                 C_Profiler *prof);

/**
 * @brief Destroy a profiler instance.
 */
C_Status insight_profiler_destroy(C_Profiler prof);

/**
 * @brief Start recording profiler events.
 */
C_Status insight_profiler_start(C_Profiler prof);

/**
 * @brief Stop recording profiler events.
 */
C_Status insight_profiler_stop(C_Profiler prof);

/**
 * @brief Clear all recorded data.
 */
C_Status insight_profiler_reset(C_Profiler prof);

/**
 * @brief Begin a named event.
 */
C_Status insight_profiler_begin_event(C_Profiler prof, const char *name);

/**
 * @brief End the current event.
 */
C_Status insight_profiler_end_event(C_Profiler prof);

/**
 * @brief Get aggregated event statistics.
 *
 * Returns pointer to internal array valid until next get_events or reset.
 */
C_Status insight_profiler_get_events(C_Profiler prof, C_ProfilerEvent **events,
                                     size_t *count);

#ifdef __cplusplus
}
#endif
