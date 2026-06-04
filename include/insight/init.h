// include/insight/init.h
#pragma once
#include "insight/core/place.h"
#include <optional>
#include <string>
#include <vector>

namespace ins {

/**
 * @brief Initialize Insight framework with smart backend discovery.
 *
 * - No args: CPU required, then auto-discover and load first GPU backend.
 * - Empty vector: load nothing.
 * - Specified backends: load each in order.
 */
void init(std::optional<std::vector<std::string>> backends = std::nullopt);

/**
 * @brief Initialize with specified backends (convenience overload).
 * Same as init(std::optional<...>(backends)).
 */
void init(const std::vector<std::string> &backends);

/**
 * @brief Check if Insight is initialized.
 */
bool is_initialized();

/**
 * @brief Check if a device type is available.
 */
bool has_device(DeviceKind kind);

/**
 * @brief Load an additional backend after init().
 * Safe to call multiple times (no-op if already loaded).
 * @param backend Backend name: "cpu", "cuda", "rocm", etc.
 */
void load_backend(const std::string &backend);

} // namespace ins
