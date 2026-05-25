#pragma once
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief Set the last error message for the current thread.
 * @param msg Null-terminated error message string (must remain valid until next
 * call)
 */
void insight_set_last_error(const char *msg);

/**
 * @brief Get the last error message for the current thread.
 * @return Null-terminated error message string
 */
const char *insight_get_last_error();

#ifdef __cplusplus
}
#endif // __cplusplus