// insight/ops/random.h
/**
 * @file random.h
 * @brief Random number generation operations.
 */

#pragma once

#include "insight/core/array.h"
#include "insight/core/place.h"
#include "insight/core/shape.h"
#include "insight/core/dtype.h"
#include <vector>
#include <string>

namespace ins {

    // ============================================================================
    // Seed Management
    // ============================================================================

    /**
     * @brief Set random seed for all available devices.
     *
     * Each device receives base_seed + device_id.
     *
     * @param base_seed Base seed value
     */
    void seed(uint64_t base_seed);

    /**
     * @brief Get current base seed.
     * @return uint64_t Current base seed
     */
    uint64_t get_seed();

    // ============================================================================
    // Basic Distributions
    // ============================================================================

    /**
     * @brief Uniform distribution in [0, 1).
     */
    Array rand(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Normal distribution (mean=0, std=1).
     */
    Array randn(const Shape& shape, DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Random integers in [low, high).
     */
    Array randint(int64_t low, int64_t high, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Normal distribution with specified parameters.
     */
    Array normal(double mean, double std, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Uniform distribution in [low, high).
     */
    Array uniform(double low, double high, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Random permutation of [0, n-1].
     */
    Array randperm(int64_t n, DType dtype = DType::I64, const Place& place = get_device());

    // ============================================================================
    // Like Functions
    // ============================================================================

    Array rand_like(const Array& x);
    Array randn_like(const Array& x);
    Array randint_like(const Array& x, int64_t low, int64_t high, DType dtype = DType::UNKNOWN);
    Array normal_like(const Array& x, double mean, double std, DType dtype = DType::UNKNOWN);
    Array uniform_like(const Array& x, double low, double high, DType dtype = DType::UNKNOWN);

    // ============================================================================
    // Additional Distributions
    // ============================================================================

    /**
     * @brief Exponential distribution.
     * @param scale Scale parameter (beta)
     */
    Array exponential(double scale, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Gamma distribution.
     * @param shape_param Shape parameter (alpha)
     * @param rate Rate parameter (beta)
     */
    Array gamma(double shape_param, double rate, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Beta distribution.
     * @param a Alpha parameter
     * @param b Beta parameter
     */
    Array beta(double a, double b, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

    /**
     * @brief Binomial distribution.
     * @param n Number of trials
     * @param p Probability of success
     */
    Array binomial(int64_t n, double p, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Poisson distribution.
     * @param lam Lambda parameter
     */
    Array poisson(double lam, const Shape& shape,
        DType dtype = DType::I64, const Place& place = get_device());

    /**
     * @brief Chi-square distribution.
     * @param df Degrees of freedom
     */
    Array chisquare(double df, const Shape& shape,
        DType dtype = DType::F32, const Place& place = get_device());

} // namespace ins