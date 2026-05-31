"""Estimation functions for signal processing.

Provides the Kalman filter for state estimation.
"""

from insight._insight import signal as _signal

__all__ = [
    "KalmanFilter",
]

#: KalmanFilter class for linear state estimation.
#:
#: Args:
#:     dim_x (int): Dimension of the state vector.
#:     dim_z (int): Dimension of the measurement vector.
#:     dim_u (int): Dimension of the control vector.  Default is 0.
#:     points (int): Number of filter points (ensemble).  Default is 1.
#:     dtype: Data type.  Default is ``float64``.
#:
#: Attributes:
#:     x: State vector.
#:     P: State covariance matrix.
#:     z: Measurement vector.
#:     R: Measurement noise covariance.
#:     Q: Process noise covariance.
#:     F: State transition matrix.
#:     H: Measurement matrix.
#:     dim_x: State dimension (read-only).
#:     dim_z: Measurement dimension (read-only).
#:     points: Number of points (read-only).
#:
#: Methods:
#:     predict(): Predict the next state.
#:     update(z): Update the state with a new measurement.
KalmanFilter = _signal.KalmanFilter
