// insight/ops/signal/waveforms.h
#pragma once
#include "insight/core/array.h"
#include <string>

namespace ins {
namespace signal {

/**
 * @brief Periodic sawtooth/triangle waveform.
 *
 * Generates a sawtooth waveform with period 2*pi.
 * For width=0.5, produces a symmetric triangle wave.
 *
 * @param t Input array (time)
 * @param width Rising segment fraction of each period (0 to 1). Default: 1.0
 * @return Sawtooth waveform array, same shape as t
 */
Array sawtooth(const Array &t, double width = 1.0);

/**
 * @brief Periodic square-wave waveform.
 *
 * Generates a square wave with period 2*pi.
 *
 * @param t Input array (time)
 * @param duty Duty cycle: fraction of each period that is +1 (0 to 1). Default:
 * 0.5
 * @return Square-wave array, same shape as t
 */
Array square(const Array &t, double duty = 0.5);

/**
 * @brief Result struct for gausspulse when multiple outputs are requested.
 */
struct GaussPulseResult {
  Array yi; ///< In-phase component
  Array yq; ///< Quadrature component (if requested)
  Array ye; ///< Envelope (if requested)
};

/**
 * @brief Gausspulse sweep signal: Gaussian-modulated sinusoid.
 *
 * Returns the in-phase component only. Use the GaussPulseResult overload
 * for quadrature/envelope outputs.
 *
 * @param t Input array (time)
 * @param fc Center frequency. Default: 1000
 * @param bw Fractional bandwidth. Default: 0.5
 * @param bwr Reference bandwidth level in dB. Default: -6
 * @param tpr Threshold for envelope truncation in dB. Default: -60
 * @return In-phase component array
 */
Array gausspulse(const Array &t, double fc = 1000, double bw = 0.5,
                 double bwr = -6, double tpr = -60);

/**
 * @brief Gausspulse with optional quadrature and envelope outputs.
 */
GaussPulseResult gausspulse_full(const Array &t, double fc = 1000,
                                 double bw = 0.5, double bwr = -6,
                                 double tpr = -60);

/**
 * @brief Chirp sweep method.
 */
enum class ChirpMethod {
  Linear,      ///< Linear frequency sweep
  Quadratic,   ///< Quadratic frequency sweep
  Logarithmic, ///< Logarithmic frequency sweep
  Hyperbolic   ///< Hyperbolic frequency sweep
};

/**
 * @brief Frequency-swept cosine generator (chirp signal).
 *
 * @param t Input array (time)
 * @param f0 Frequency at time 0
 * @param t1 Time at which f1 is attained
 * @param f1 Frequency at time t1
 * @param method Sweep method. Default: Linear
 * @param phi Phase offset in radians. Default: 0
 * @param vertex_zero If true (default), frequency sweep starts at f0
 * @return Chirp signal array
 */
Array chirp(const Array &t, double f0, double t1, double f1,
            ChirpMethod method = ChirpMethod::Linear, double phi = 0.0,
            bool vertex_zero = true);

/**
 * @brief Discrete unit impulse (Kronecker delta).
 *
 * Creates an array of zeros with a single 1 at the specified index.
 *
 * @param shape Shape of the output array
 * @param idx Index of the impulse. If -1, places at center. Default: -1
 * @param dtype Data type. Default: F64
 * @param place Device placement. Default: current device
 * @return Array with a single 1 at idx, zeros elsewhere
 */
Array unit_impulse(const Shape &shape, int64_t idx = -1,
                   DType dtype = DType::F64, const Place &place = get_device());

} // namespace signal
} // namespace ins
