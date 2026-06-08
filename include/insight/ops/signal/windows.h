// insight/ops/signal/windows.h
#pragma once
#include "insight/core/array.h"
#include <string>
#include <vector>

namespace ins {
namespace signal {

// ============================================================================
// Helper / General
// ============================================================================

/**
 * @brief Generic weighted sum of cosine terms window.
 *
 * w[n] = sum_{k=0}^{len(a)-1} (-1)^k * a[k] * cos(2*pi*k*n / (M-1))
 *
 * @param M Window length (number of samples)
 * @param a Coefficients for the cosine sum
 * @param sym If true, generate a symmetric window; if false, DFT-even
 * (periodic)
 * @return 1D Array of length M
 */
Array general_cosine(int64_t M, const std::vector<double> &a, bool sym = true);

/**
 * @brief Factory function: get a window by name.
 *
 * Accepts standard window names: "boxcar", "triang", "hann", "hamming",
 * "blackman", "kaiser", "gaussian", etc.
 *
 * @param window Window name string
 * @param Nx Window length
 * @param fftbins If true, DFT-even (periodic) window
 * @return 1D Array of length Nx
 */
Array get_window(const std::string &window, int64_t Nx, bool fftbins = true);

/**
 * @brief Get a window that requires one numeric parameter.
 *
 * For windows like kaiser (beta), gaussian (std), general_hamming (alpha).
 *
 * @param window Window name string
 * @param param The numeric parameter (e.g., Kaiser beta, Gaussian std)
 * @param Nx Window length
 * @param fftbins If true, DFT-even (periodic) window
 * @return 1D Array of length Nx
 */
Array get_window(const std::string &window, double param, int64_t Nx,
                 bool fftbins = true);

// ============================================================================
// Simple Windows
// ============================================================================

/**
 * @brief Rectangular (boxcar/Dirichlet) window: all ones.
 */
Array boxcar(int64_t M, bool sym = true);

/**
 * @brief Triangular window.
 */
Array triang(int64_t M, bool sym = true);

/**
 * @brief Parzen window (piecewise polynomial).
 */
Array parzen(int64_t M, bool sym = true);

/**
 * @brief Bohman window (product of cosine and triangular).
 */
Array bohman(int64_t M, bool sym = true);

/**
 * @brief Bartlett window (triangular with zero endpoints).
 */
Array bartlett(int64_t M, bool sym = true);

/**
 * @brief Cosine (sine) window.
 */
Array cosine(int64_t M, bool sym = true);

/**
 * @brief Exponential (Poisson) window.
 *
 * @param M Window length
 * @param center Center of the exponential decay. Default: (M-1)/2
 * @param tau Decay constant. Default: 1.0
 * @param sym Symmetry flag
 */
Array exponential(int64_t M, double center = -1.0, double tau = 1.0,
                  bool sym = true);

// ============================================================================
// Cosine-Sum Windows (via general_cosine)
// ============================================================================

/**
 * @brief Blackman window: 0.42 - 0.5*cos(2*pi*n/(M-1)) + 0.08*cos(4*pi*n/(M-1))
 */
Array blackman(int64_t M, bool sym = true);

/**
 * @brief Minimum 4-term Blackman-Harris (Nuttall) window.
 */
Array nuttall(int64_t M, bool sym = true);

/**
 * @brief Minimum 4-term Blackman-Harris window.
 */
Array blackmanharris(int64_t M, bool sym = true);

/**
 * @brief Flat-top window (5-term cosine sum, maximally flat main lobe).
 */
Array flattop(int64_t M, bool sym = true);

/**
 * @brief Hann (Hanning) window: 0.5 - 0.5*cos(2*pi*n/(M-1))
 */
Array hann(int64_t M, bool sym = true);

/**
 * @brief Generalized Hamming window: alpha - (1-alpha)*cos(2*pi*n/(M-1))
 */
Array general_hamming(int64_t M, double alpha, bool sym = true);

/**
 * @brief Hamming window: 0.54 - 0.46*cos(2*pi*n/(M-1))
 */
Array hamming(int64_t M, bool sym = true);

// ============================================================================
// Parameterized Windows
// ============================================================================

/**
 * @brief Tukey (tapered cosine) window.
 *
 * @param M Window length
 * @param alpha Taper fraction (0 = rectangular, 1 = Hann). Default: 0.5
 * @param sym Symmetry flag
 */
Array tukey(int64_t M, double alpha = 0.5, bool sym = true);

/**
 * @brief Modified Bartlett-Hann window.
 */
Array barthann(int64_t M, bool sym = true);

/**
 * @brief Kaiser window (Bessel-function based).
 *
 * @param M Window length
 * @param beta Kaiser shape parameter
 * @param sym Symmetry flag
 */
Array kaiser(int64_t M, double beta, bool sym = true);

/**
 * @brief Gaussian window.
 *
 * @param M Window length
 * @param std Standard deviation (in samples)
 * @param sym Symmetry flag
 */
Array gaussian(int64_t M, double std, bool sym = true);

/**
 * @brief Generalized Gaussian window.
 *
 * @param M Window length
 * @param p Shape parameter (p=1 is Gaussian, p=2 is super-Gaussian)
 * @param sig Standard deviation (in samples)
 * @param sym Symmetry flag
 */
Array general_gaussian(int64_t M, double p, double sig, bool sym = true);

/**
 * @brief Dolph-Chebyshev window.
 *
 * @param M Window length
 * @param at Attenuation in dB (side lobe level)
 * @param sym Symmetry flag
 */
Array chebwin(int64_t M, double at, bool sym = true);

/**
 * @brief Taylor window.
 *
 * @param M Window length
 * @param nbar Number of nearly constant side lobes. Default: 4
 * @param sll Desired side lobe level in dB (negative). Default: -30
 * @param norm If true, normalize so maximum is 1. Default: true
 * @param sym Symmetry flag
 */
Array taylor(int64_t M, int64_t nbar = 4, double sll = -30.0, bool norm = true,
             bool sym = true);

// ============================================================================
// Quadrature Mirror Filter
// ============================================================================

/**
 * @brief Returns a Quadrature Mirror Filter (QMF) pair.
 *
 * Constructs the lowpass and highpass filters for a quadrature mirror
 * filter bank. The highpass filter is a frequency-shifted and
 * time-reversed version of the lowpass filter.
 *
 * @param h_low Lowpass filter coefficients (1D)
 * @return Pair of {lowpass, highpass} filter arrays
 */
std::pair<Array, Array> qmf(const Array &h_low);

} // namespace signal
} // namespace ins
