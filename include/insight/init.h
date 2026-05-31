// include/insight/init.h
#pragma once
#include "insight/core/place.h"
#include <string>
#include <vector>

namespace ins {

/**
 * @brief Initialize Insight framework
 * @param backends List of backends to load
 *                 "cpu" - CPU backend
 *                 any other string (e.g., "cuda", "rocm") - GPU backend
 */
void init(const std::vector<std::string> &backends = {"cpu"});

/**
 * @brief Check if Insight is initialized.
 *
 * @return true if initialized, false otherwise.
 */
bool is_initialized();

/**
 * @brief Check if a device type is available.
 * @param kind Device kind to check
 * @return true if the device backend is loaded and available
 */
bool has_device(DeviceKind kind);

/**
 * @brief Load an additional backend after init().
 *
 * Useful for loading GPU backends after the framework is already initialized
 * with CPU only. Safe to call multiple times (no-op if already loaded).
 *
 * @param backend Backend name: "cpu", "cuda", "rocm", etc.
 */
void load_backend(const std::string &backend);

} // namespace ins