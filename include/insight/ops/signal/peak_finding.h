// insight/ops/signal/peak_finding.h
#pragma once
#include "insight/core/array.h"
#include <string>
#include <vector>

namespace ins {
namespace signal {

/**
 * @brief Find indices of relative extrema in data.
 *
 * A point is a relative extremum if it is the maximum (or minimum) of
 * `order` neighbors on each side along the specified axis.
 *
 * @param data Input array
 * @param comparator Comparison direction: "greater" for maxima, "less" for
 *                   minima
 * @param axis Axis along which to find extrema. Default: 0
 * @param order Number of neighbors on each side. Default: 1
 * @param mode Edge handling: "clip" or "wrap". Default: "clip"
 * @return Vector of index arrays (one per dimension)
 */
std::vector<Array> argrelextrema(const Array &data,
                                 const std::string &comparator, int axis = 0,
                                 int order = 1,
                                 const std::string &mode = "clip");

/**
 * @brief Find indices of relative maxima in data.
 *
 * Convenience wrapper around argrelextrema with comparator="greater".
 *
 * @param data Input array
 * @param axis Axis along which to find maxima. Default: 0
 * @param order Number of neighbors on each side. Default: 1
 * @param mode Edge handling. Default: "clip"
 * @return Vector of index arrays
 */
std::vector<Array> argrelmax(const Array &data, int axis = 0, int order = 1,
                             const std::string &mode = "clip");

/**
 * @brief Find indices of relative minima in data.
 *
 * Convenience wrapper around argrelextrema with comparator="less".
 *
 * @param data Input array
 * @param axis Axis along which to find minima. Default: 0
 * @param order Number of neighbors on each side. Default: 1
 * @param mode Edge handling. Default: "clip"
 * @return Vector of index arrays
 */
std::vector<Array> argrelmin(const Array &data, int axis = 0, int order = 1,
                             const std::string &mode = "clip");

} // namespace signal
} // namespace ins
