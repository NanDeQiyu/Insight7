// insight/ops/signal/estimation.h
#pragma once
#include "insight/core/array.h"
#include "insight/core/dtype.h"

namespace ins {
namespace signal {

/**
 * @brief Multi-point Kalman Filter.
 *
 * Tracks multiple independent Kalman filters simultaneously.
 * All matrices are stacked on the points dimension for batched processing.
 *
 * Based on filterpy's KalmanFilter with GPU acceleration via composite ops.
 */
class KalmanFilter {
public:
  // State
  Array x; ///< State estimate [points, dim_x, 1]
  Array P; ///< State covariance [points, dim_x, dim_x]

  // Measurement
  Array z; ///< Last measurement [points, dim_z, 1]
  Array R; ///< Measurement noise [points, dim_z, dim_z]

  // Process
  Array Q; ///< Process noise [points, dim_x, dim_x]
  Array F; ///< State transition [points, dim_x, dim_x]
  Array H; ///< Measurement function [points, dim_z, dim_x]

  // Dimensions
  int dim_x;  ///< Number of state variables
  int dim_z;  ///< Number of measurement inputs
  int dim_u;  ///< Number of control inputs
  int points; ///< Number of filter points

  /**
   * @brief Construct a KalmanFilter.
   *
   * @param dim_x Number of state variables
   * @param dim_z Number of measurement inputs
   * @param dim_u Number of control inputs. Default: 0
   * @param points Number of filter points. Default: 1
   * @param dtype Data type (F32 or F64). Default: F64
   */
  KalmanFilter(int dim_x, int dim_z, int dim_u = 0, int points = 1,
               DType dtype = DType::F64);

  /**
   * @brief Predict step.
   *
   * x = F * x
   * P = F * P * F^T + Q
   */
  void predict();

  /**
   * @brief Update step.
   *
   * @param z Measurement [points, dim_z, 1]
   */
  void update(const Array &z);

private:
  DType dtype_;
  Place place_;
};

} // namespace signal
} // namespace ins
