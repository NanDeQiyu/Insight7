#pragma once

static thread_local const char *last_error_message = nullptr;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void insight_set_last_error(const char *msg) { last_error_message = msg; }

/**
 * @brief Get the last error message for the current thread.
 * @return Null-terminated error message string
 */
const char *insight_get_last_error() {
  return last_error_message ? last_error_message : "No error message set";
}

#ifdef __cplusplus
}
#endif // __cplusplus