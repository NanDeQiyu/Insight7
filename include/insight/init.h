// include/insight/init.h
#pragma once
#include <vector>
#include <string>
#include "insight/core/place.h"

namespace ins {

	/**
	 * @brief Initialize Insight framework
	 * @param backends List of backends to load
	 *                 "cpu" - CPU backend
	 *                 any other string (e.g., "cuda", "rocm") - GPU backend
	 */
	void init(const std::vector<std::string>& backends = { "cpu" });

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

} // namespace ins