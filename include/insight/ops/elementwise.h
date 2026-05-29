// include/insight/ops/elementwise.h
/**
 * @file elementwise.h
 * @brief Binary elementwise operations on Insight arrays.
 *
 * All functions in this file operate elementwise on two input arrays
 * (with broadcasting) and return a new array.
 *
 * @see unary.h for unary elementwise operations
 * @see reduction.h for reduction operations
 */
#pragma once
#include "insight/core/array.h"

// Backward compatibility: unary functions moved to unary.h
#include "insight/ops/unary.h"

namespace ins {

// ============================================================================
// Arithmetic operations (binary)
// ============================================================================

/** @brief Elementwise addition (a + b). Supports broadcasting. */
Array add(const Array &a, const Array &b);
/** @brief Elementwise subtraction (a - b). Supports broadcasting. */
Array sub(const Array &a, const Array &b);
/** @brief Elementwise multiplication (a * b). Supports broadcasting. */
Array mul(const Array &a, const Array &b);
/** @brief Elementwise division (a / b). Supports broadcasting. */
Array div(const Array &a, const Array &b);
/** @brief Elementwise power (a ^ b). Supports broadcasting. */
Array pow(const Array &a, const Array &b);
/** @brief Elementwise modulo (a % b). Supports broadcasting. */
Array mod(const Array &a, const Array &b);

// ============================================================================
// Comparison operations (binary)
// ============================================================================

/** @brief Elementwise equality (a == b). Returns bool array. */
Array equal(const Array &a, const Array &b);
/** @brief Elementwise inequality (a != b). Returns bool array. */
Array not_equal(const Array &a, const Array &b);
/** @brief Elementwise greater-than (a > b). Returns bool array. */
Array greater(const Array &a, const Array &b);
/** @brief Alias for greater(). */
Array greater_than(const Array &a, const Array &b);
/** @brief Elementwise less-than (a < b). Returns bool array. */
Array less(const Array &a, const Array &b);
/** @brief Alias for less(). */
Array less_than(const Array &a, const Array &b);
/** @brief Elementwise greater-or-equal (a >= b). Returns bool array. */
Array greater_equal(const Array &a, const Array &b);
/** @brief Elementwise less-or-equal (a <= b). Returns bool array. */
Array less_equal(const Array &a, const Array &b);

// ============================================================================
// Logical operations (binary)
// ============================================================================

/** @brief Elementwise logical AND. Returns bool array. */
Array logical_and(const Array &a, const Array &b);
/** @brief Elementwise logical OR. Returns bool array. */
Array logical_or(const Array &a, const Array &b);
/** @brief Elementwise logical XOR. Returns bool array. */
Array logical_xor(const Array &a, const Array &b);

// ============================================================================
// Bitwise operations (binary, integer types only)
// ============================================================================

/** @brief Elementwise bitwise AND. */
Array bitwise_and(const Array &a, const Array &b);
/** @brief Elementwise bitwise OR. */
Array bitwise_or(const Array &a, const Array &b);
/** @brief Elementwise bitwise XOR. */
Array bitwise_xor(const Array &a, const Array &b);
/** @brief Elementwise bitwise left shift. */
Array bitwise_left_shift(const Array &a, const Array &b);
/** @brief Elementwise bitwise right shift. */
Array bitwise_right_shift(const Array &a, const Array &b);

// ============================================================================
// Maximum / Minimum
// ============================================================================

/** @brief Elementwise maximum of two arrays. */
Array maximum(const Array &a, const Array &b);
/** @brief Elementwise minimum of two arrays. */
Array minimum(const Array &a, const Array &b);

} // namespace ins
