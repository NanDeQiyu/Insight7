// insight/ops/signal/radar.h
#pragma once
#include "insight/core/array.h"
#include <string>

namespace ins {
namespace signal {

/**
 * @brief Pulse compression (matched filtering).
 *
 * Performs matched filtering of received signal x with transmitted template.
 *
 * @param x Received signal, 2D [num_pulses, samples_per_pulse]
 * @param template_tx Transmitted signal, 1D
 * @param normalize Normalize the template. Default: false
 * @param window Window name or empty. Default: empty (no window)
 * @param nfft FFT size. Default: samples_per_pulse
 * @return Compressed IQ (complex)
 */
Array pulse_compression(const Array &x, const Array &template_tx,
                        bool normalize = false, const std::string &window = "",
                        int64_t nfft = 0);

/**
 * @brief Pulse Doppler processing.
 *
 * FFT across slow-time (pulse) dimension to yield range/doppler matrix.
 *
 * @param x Received signal, 2D [num_pulses, samples_per_pulse]
 * @param window Window name or empty. Default: empty
 * @param nfft FFT size. Default: num_pulses
 * @return Range/Doppler matrix (complex)
 */
Array pulse_doppler(const Array &x, const std::string &window = "",
                    int64_t nfft = 0);

/**
 * @brief Compute CFAR alpha factor.
 *
 * alpha = N * (pfa^(-1/N) - 1)
 *
 * @param pfa Probability of false alarm
 * @param N Number of reference cells
 * @return Alpha factor
 */
double cfar_alpha(double pfa, int N);

/**
 * @brief Cell-Averaging CFAR detector.
 *
 * Detects targets in 1D or 2D data using cell-averaging CFAR.
 *
 * @param data Input array (1D or 2D)
 * @param guard_cells Number of guard cells per side (1D: scalar, 2D:
 *                    {row, col})
 * @param reference_cells Number of reference cells per side
 * @param pfa Probability of false alarm. Default: 1e-3
 * @return Pair of {threshold, detections} arrays
 */
std::pair<Array, Array> ca_cfar(const Array &data,
                                const std::vector<int> &guard_cells,
                                const std::vector<int> &reference_cells,
                                double pfa = 1e-3);

/**
 * @brief MVDR beamformer weights.
 *
 * Minimum variance distortionless response beamformer.
 *
 * @param x Received signal, 2D [num_sensors, num_samples]
 * @param sv Steering vector, 1D [num_sensors]
 * @param calc_cov Whether to compute covariance from x. Default: true
 * @return Weight vector
 */
Array mvdr(const Array &x, const Array &sv, bool calc_cov = true);

/**
 * @brief Normalized ambiguity function.
 *
 * @param x Input signal (1D, complex)
 * @param fs Sampling frequency
 * @param prf Pulse repetition frequency
 * @param y Optional second signal. Default: same as x
 * @param cut Cut type: "2d", "delay", or "doppler". Default: "2d"
 * @param cutValue Cut value for delay/doppler mode. Default: 0
 * @return Ambiguity function result
 */
Array ambgfun(const Array &x, double fs, double prf, const Array &y = Array(),
              const std::string &cut = "2d", double cutValue = 0);

} // namespace signal
} // namespace ins
