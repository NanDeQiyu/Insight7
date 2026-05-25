// insight/ops/cast.h
#pragma once
#include "insight/core/array.h"
#include "insight/core/dtype.h"

namespace ins {

/**
 * @brief Cast an array to a different data type (same device).
 *
 * Converts each element of the input array to the target data type.
 * The shape and device of the output array are the same as the input.
 *
 * @param input        Input array
 * @param target_dtype Target data type
 * @param copy         If false and dtype already matches, return input itself.
 *                     If true, always return a copy (default: true)
 * @return Array with target dtype, same shape and device as input
 */
Array cast(const Array &input, DType target_dtype, bool copy = true);

/**
 * @brief Cast an array to match another array's data type.
 *
 * @param input Input array
 * @param other Array whose dtype is used as target
 * @param copy  If false and already matching, return input itself (default:
 * true)
 * @return Array cast to other.dtype(), same shape and device as input
 */
Array cast_like(const Array &input, const Array &other, bool copy = true);

} // namespace ins