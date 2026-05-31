// insight/ops/signal/spectral_analysis.h
#pragma once
#include "insight/core/array.h"
#include <string>
#include <vector>

namespace ins {
namespace signal {

// ============================================================================
// Spectral Estimation
// ============================================================================

/// @brief Result struct for functions returning (f, Pxx) or (f, Pxy)
struct SpectralResult {
  Array f;   ///< Frequency array
  Array Pxx; ///< Power/cross spectral density
};

/// @brief Result struct for spectrogram/stft returning (f, t, Sxx)
struct SpectrogramResult {
  Array f;   ///< Frequency array
  Array t;   ///< Time array
  Array Sxx; ///< Spectrogram matrix
};

/// @brief Estimate cross power spectral density using Welch's method.
///
/// @param x First input signal (1D)
/// @param y Second input signal (1D)
/// @param fs Sampling frequency. Default: 1.0
/// @param window Window name. Default: "hann"
/// @param nperseg Segment length. Default: 256
/// @param noverlap Overlap between segments. Default: nperseg/2
/// @param nfft FFT length. Default: nperseg
/// @param detrend Detrending method: "constant" or "none". Default: "constant"
/// @param return_onesided One-sided spectrum for real data. Default: true
/// @param scaling "density" (V²/Hz) or "spectrum" (V²). Default: "density"
/// @return SpectralResult with f (frequencies) and Pxy (cross spectral density)
SpectralResult csd(const Array &x, const Array &y, double fs = 1.0,
                   const std::string &window = "hann", int64_t nperseg = 256,
                   int64_t noverlap = 0, int64_t nfft = 0,
                   const std::string &detrend = "constant",
                   bool return_onesided = true,
                   const std::string &scaling = "density");

/// @brief Estimate power spectral density using Welch's method.
///
/// @param x Input signal (1D)
/// @param fs Sampling frequency. Default: 1.0
/// @param window Window name. Default: "hann"
/// @param nperseg Segment length. Default: 256
/// @param noverlap Overlap. Default: nperseg/2
/// @param nfft FFT length. Default: nperseg
/// @param detrend Detrending method. Default: "constant"
/// @param return_onesided One-sided spectrum. Default: true
/// @param scaling "density" or "spectrum". Default: "density"
/// @return SpectralResult with f and Pxx
SpectralResult welch(const Array &x, double fs = 1.0,
                     const std::string &window = "hann", int64_t nperseg = 256,
                     int64_t noverlap = 0, int64_t nfft = 0,
                     const std::string &detrend = "constant",
                     bool return_onesided = true,
                     const std::string &scaling = "density");

/// @brief Estimate power spectral density using a periodogram.
///
/// Convenience wrapper around welch with noverlap=0.
SpectralResult periodogram(const Array &x, double fs = 1.0,
                           const std::string &window = "boxcar",
                           int64_t nfft = 0,
                           const std::string &detrend = "constant",
                           bool return_onesided = true,
                           const std::string &scaling = "density");

/// @brief Estimate magnitude squared coherence using Welch's method.
///
/// Cxy = |Pxy|² / (Pxx * Pyy)
SpectralResult coherence(const Array &x, const Array &y, double fs = 1.0,
                         const std::string &window = "hann",
                         int64_t nperseg = 256, int64_t noverlap = 0,
                         int64_t nfft = 0,
                         const std::string &detrend = "constant");

/// @brief Compute a spectrogram (time-frequency representation).
SpectrogramResult
spectrogram(const Array &x, double fs = 1.0, const std::string &window = "hann",
            int64_t nperseg = 256, int64_t noverlap = 0, int64_t nfft = 0,
            const std::string &detrend = "constant",
            bool return_onesided = true, const std::string &mode = "psd");

/// @brief Compute the Short-Time Fourier Transform (STFT).
SpectrogramResult stft(const Array &x, double fs = 1.0,
                       const std::string &window = "hann",
                       int64_t nperseg = 256, int64_t noverlap = 0,
                       int64_t nfft = 0);

// ============================================================================
// Phase Synchrony
// ============================================================================

/// @brief Determine vector strength (phase synchrony).
///
/// @param events 1D array of event times
/// @param period Period of the signal (or array of periods)
/// @return Pair of {strength, phase}
std::pair<double, double> vectorstrength(const Array &events, double period);

} // namespace signal
} // namespace ins
