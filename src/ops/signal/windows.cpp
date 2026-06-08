// src/ops/signal/windows.cpp
#include "insight/ops/signal/windows.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/indexing.h"
#include "insight/ops/manipulation.h"
#include "insight/ops/reduction.h"
#include "insight/ops/unary.h"
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ins {
namespace signal {

namespace {

// Helper: if sym=false, compute window of size M+1 and truncate to M.
// `fn` receives M_work and returns an Array of size M_work.
template <typename Fn> Array apply_sym(int64_t M, bool sym, Fn &&fn) {
  INS_CHECK(M >= 1, "window: M must be >= 1");
  if (M == 1) {
    return full({1}, 1.0, DType::F64);
  }
  int64_t M_work = sym ? M : M + 1;
  Array result = fn(M_work);
  if (!sym) {
    result = slice(result, {0}, {0}, {M});
  }
  return result;
}

// Helper: compute standard cosine-sum window
// w[n] = a[0] + sum_{k=1}^{K-1} (-1)^k * a[k] * cos(2*pi*k*n/(N-1))
Array cosine_sum_window(int64_t N, const std::vector<double> &a) {
  double den = N - 1.0;
  Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);

  // Start with a[0]
  Array result = full({N}, a[0], DType::F64);

  for (size_t k = 1; k < a.size(); ++k) {
    double sign = (k % 2 == 0) ? 1.0 : -1.0;
    // phase = 2*pi*k*n / (N-1)
    Array phase = mul(i_arr, full({N}, 2.0 * M_PI * k / den, DType::F64));
    Array term = mul(cos(phase), full({N}, sign * a[k], DType::F64));
    result = add(result, term);
  }
  return result;
}

// Modified Bessel function I0(x) — for Kaiser window
// Computes I0 on a scalar (runs on CPU, result transferred via full())
double bessel_i0(double x) {
  double ax = std::abs(x);
  if (ax < 3.75) {
    double t = x / 3.75;
    double t2 = t * t;
    return 1.0 + t2 * (3.5156229 +
                       t2 * (3.0899424 +
                             t2 * (1.2067492 +
                                   t2 * (0.2659732 +
                                         t2 * (0.0360768 + t2 * 0.0045813)))));
  }
  double t = 3.75 / ax;
  return (std::exp(ax) / std::sqrt(ax)) *
         (0.39894228 +
          t * (0.01328592 +
               t * (0.00225319 +
                    t * (-0.00157565 +
                         t * (0.00916281 +
                              t * (-0.02057706 +
                                   t * (0.02635537 +
                                        t * (-0.01647633 +
                                             t * 0.00392377))))))));
}

} // anonymous namespace

// ============================================================================
// general_cosine
// ============================================================================

Array general_cosine(int64_t M, const std::vector<double> &a, bool sym) {
  INS_CHECK(M >= 1, "general_cosine: M must be >= 1");
  INS_CHECK(!a.empty(), "general_cosine: coefficients must not be empty");
  return apply_sym(M, sym, [&](int64_t N) { return cosine_sum_window(N, a); });
}

// ============================================================================
// boxcar
// ============================================================================

Array boxcar(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) { return ones({N}, DType::F64); });
}

// ============================================================================
// triang
// ============================================================================

Array triang(int64_t M, bool sym) {
  // Compute on CPU (mirror logic is tricky for composite ops)
  return apply_sym(M, sym, [](int64_t N) {
    std::vector<double> vals(N);
    bool odd = (N % 2 == 1);
    for (int64_t i = 0; i < N; ++i) {
      int64_t n = i + 1; // 1-based
      if (n <= (N + 1) / 2) {
        vals[i] = odd ? (2.0 * n / (N + 1.0)) : ((2.0 * n - 1.0) / N);
      } else {
        vals[i] = vals[N - 1 - i];
      }
    }
    return to_array(vals);
  });
}

// ============================================================================
// parzen
// ============================================================================

Array parzen(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    double half_N = N / 2.0;
    double quarter = (N - 1) / 4.0;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // n = i - (N-1)/2
    Array n = sub(i_arr, full({N}, (N - 1.0) / 2.0, DType::F64));
    Array an = abs(n);

    // Region: |n| > quarter → 2*(1 - |n|/(N/2))^3
    Array u =
        sub(full({N}, 1.0, DType::F64), div(an, full({N}, half_N, DType::F64)));
    Array u2 = mul(u, u);
    Array u3 = mul(u2, u);
    Array tail = mul(full({N}, 2.0, DType::F64), u3);

    // Region: |n| <= quarter → 1 - 6*t^2 + 6*t^3 where t = |n|/(N/2)
    Array t = div(an, full({N}, half_N, DType::F64));
    Array t2 = mul(t, t);
    Array t3 = mul(t2, t);
    Array mid = add(
        sub(full({N}, 1.0, DType::F64), mul(full({N}, 6.0, DType::F64), t2)),
        mul(full({N}, 6.0, DType::F64), t3));

    Array mask = greater_than(an, full({N}, quarter, DType::F64));
    return where(mask, tail, mid);
  });
}

// ============================================================================
// bohman
// ============================================================================

Array bohman(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    double den = N - 1.0;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // fac = |i/(N-1) - 0.5| * 2
    Array frac =
        sub(div(i_arr, full({N}, den, DType::F64)), full({N}, 0.5, DType::F64));
    Array fac = mul(abs(frac), full({N}, 2.0, DType::F64));

    // w = (1-fac)*cos(pi*fac) + (1/pi)*sin(pi*fac)  for fac < 1
    Array pi_fac = mul(fac, full({N}, M_PI, DType::F64));
    Array one_minus_fac = sub(full({N}, 1.0, DType::F64), fac);
    Array w_val = add(mul(one_minus_fac, cos(pi_fac)),
                      mul(full({N}, 1.0 / M_PI, DType::F64), sin(pi_fac)));

    Array mask = less_than(fac, full({N}, 1.0, DType::F64));
    Array zero = full({N}, 0.0, DType::F64);
    return where(mask, w_val, zero);
  });
}

// ============================================================================
// bartlett
// ============================================================================

Array bartlett(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    double den = N - 1.0;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    Array ramp =
        div(mul(i_arr, full({N}, 2.0, DType::F64)), full({N}, den, DType::F64));
    Array anti_ramp = sub(full({N}, 2.0, DType::F64), ramp);

    Array mask =
        less_equal(i_arr, full({N}, static_cast<double>(N / 2), DType::F64));
    return where(mask, ramp, anti_ramp);
  });
}

// ============================================================================
// cosine
// ============================================================================

Array cosine(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // w = sin(pi/N * (i + 0.5))
    Array phase = mul(add(i_arr, full({N}, 0.5, DType::F64)),
                      full({N}, M_PI / N, DType::F64));
    return sin(phase);
  });
}

// ============================================================================
// exponential
// ============================================================================

Array exponential(int64_t M, double center, double tau, bool sym) {
  return apply_sym(M, sym, [&](int64_t N) {
    double c = (center < 0) ? (N - 1.0) / 2.0 : center;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // w = exp(-|i - center| / tau)
    Array diff = abs(sub(i_arr, full({N}, c, DType::F64)));
    return exp(div(negative(diff), full({N}, tau, DType::F64)));
  });
}

// ============================================================================
// blackman
// ============================================================================

Array blackman(int64_t M, bool sym) {
  return general_cosine(M, {0.42, 0.50, 0.08}, sym);
}

// ============================================================================
// nuttall
// ============================================================================

Array nuttall(int64_t M, bool sym) {
  return general_cosine(M, {0.3635819, 0.4891775, 0.1365995, 0.0106411}, sym);
}

// ============================================================================
// blackmanharris
// ============================================================================

Array blackmanharris(int64_t M, bool sym) {
  return general_cosine(M, {0.35875, 0.48829, 0.14128, 0.01168}, sym);
}

// ============================================================================
// flattop
// ============================================================================

Array flattop(int64_t M, bool sym) {
  return general_cosine(
      M, {0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368}, sym);
}

// ============================================================================
// hann
// ============================================================================

Array hann(int64_t M, bool sym) { return general_hamming(M, 0.5, sym); }

// ============================================================================
// general_hamming
// ============================================================================

Array general_hamming(int64_t M, double alpha, bool sym) {
  return general_cosine(M, {alpha, 1.0 - alpha}, sym);
}

// ============================================================================
// hamming
// ============================================================================

Array hamming(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // w = 0.54 - 0.46*cos(2*pi*i/(N-1))
    Array phase = mul(i_arr, full({N}, 2.0 * M_PI / (N - 1), DType::F64));
    return add(full({N}, 0.54, DType::F64),
               mul(full({N}, -0.46, DType::F64), cos(phase)));
  });
}

// ============================================================================
// tukey
// ============================================================================

Array tukey(int64_t M, double alpha, bool sym) {
  INS_CHECK(alpha >= 0.0 && alpha <= 1.0,
            "tukey: alpha must be in [0, 1], got ", alpha);
  return apply_sym(M, sym, [&](int64_t N) {
    if (alpha <= 0.0) {
      return ones({N}, DType::F64);
    }
    if (alpha >= 1.0) {
      // Hann
      Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
      Array phase = mul(i_arr, full({N}, 2.0 * M_PI / (N - 1), DType::F64));
      return add(full({N}, 0.5, DType::F64),
                 mul(full({N}, -0.5, DType::F64), cos(phase)));
    }

    double width = alpha * (N - 1) / 2.0;
    double den = N - 1.0;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);

    // Left taper: i < width
    Array left_phase = mul(div(i_arr, full({N}, width, DType::F64)),
                           full({N}, M_PI, DType::F64));
    left_phase = sub(left_phase, full({N}, M_PI, DType::F64));
    Array left = mul(full({N}, 0.5, DType::F64),
                     add(full({N}, 1.0, DType::F64), cos(left_phase)));

    // Right taper: i >= den - width
    Array right_phase = mul(div(i_arr, full({N}, width, DType::F64)),
                            full({N}, M_PI, DType::F64));
    right_phase =
        sub(right_phase, full({N}, M_PI * (2.0 / alpha - 1.0), DType::F64));
    Array right = mul(full({N}, 0.5, DType::F64),
                      add(full({N}, 1.0, DType::F64), cos(right_phase)));

    // Middle: flat 1
    Array middle = ones({N}, DType::F64);

    // Combine: left | middle | right
    Array left_mask = less_than(i_arr, full({N}, width, DType::F64));
    Array right_mask = greater_equal(i_arr, full({N}, den - width, DType::F64));

    return where(left_mask, left, where(right_mask, right, middle));
  });
}

// ============================================================================
// barthann
// ============================================================================

Array barthann(int64_t M, bool sym) {
  return apply_sym(M, sym, [](int64_t N) {
    double den = N - 1.0;
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // frac = i/(N-1) - 0.5
    Array frac =
        sub(div(i_arr, full({N}, den, DType::F64)), full({N}, 0.5, DType::F64));
    Array abs_frac = abs(frac);
    // w = 0.62 - 0.48*|frac| + 0.38*cos(2*pi*|frac|)
    Array cos_term = cos(mul(abs_frac, full({N}, 2.0 * M_PI, DType::F64)));
    return add(add(full({N}, 0.62, DType::F64),
                   mul(full({N}, -0.48, DType::F64), abs_frac)),
               mul(full({N}, 0.38, DType::F64), cos_term));
  });
}

// ============================================================================
// kaiser
// ============================================================================

Array kaiser(int64_t M, double beta, bool sym) {
  return apply_sym(M, sym, [&](int64_t N) {
    double alpha_2 = (N - 1.0) / 2.0;
    double i0_beta = bessel_i0(beta);
    // Pre-compute on CPU and transfer via full()
    // w[i] = I0(beta * sqrt(1 - ((i-alpha_2)/alpha_2)^2)) / I0(beta)
    // We compute the values on CPU and create the array
    std::vector<double> vals(N);
    for (int64_t i = 0; i < N; ++i) {
      double ratio = (i - alpha_2) / alpha_2;
      vals[i] = bessel_i0(beta * std::sqrt(1.0 - ratio * ratio)) / i0_beta;
    }
    return to_array(vals);
  });
}

// ============================================================================
// gaussian
// ============================================================================

Array gaussian(int64_t M, double std, bool sym) {
  INS_CHECK(std > 0, "gaussian: std must be > 0");
  return apply_sym(M, sym, [&](int64_t N) {
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    // n = i - (N-1)/2
    Array n = sub(i_arr, full({N}, (N - 1.0) / 2.0, DType::F64));
    // w = exp(-0.5 * (n/std)^2)
    Array ratio = div(n, full({N}, std, DType::F64));
    Array ratio2 = mul(ratio, ratio);
    return exp(mul(ratio2, full({N}, -0.5, DType::F64)));
  });
}

// ============================================================================
// general_gaussian
// ============================================================================

Array general_gaussian(int64_t M, double p, double sig, bool sym) {
  INS_CHECK(sig > 0, "general_gaussian: sig must be > 0");
  return apply_sym(M, sym, [&](int64_t N) {
    Array i_arr = arange(0.0, static_cast<double>(N), 1.0, DType::F64);
    Array n = sub(i_arr, full({N}, (N - 1.0) / 2.0, DType::F64));
    // w = exp(-0.5 * |n/sig|^(2*p))
    Array ratio = div(abs(n), full({N}, sig, DType::F64));
    // |n/sig|^(2*p) via exp(2*p*ln(|n/sig|))
    // But ln(0) is -inf, so handle via pow
    // Use: exp(-0.5 * pow(|n/sig|, 2*p))
    // pow is not in our ops, so compute via exp and log
    // Actually, let's compute on CPU for simplicity (Bessel-like)
    std::vector<double> vals(N);
    double center = (N - 1.0) / 2.0;
    for (int64_t i = 0; i < N; ++i) {
      double ni = i - center;
      vals[i] = std::exp(-0.5 * std::pow(std::abs(ni / sig), 2.0 * p));
    }
    return to_array(vals);
  });
}

// ============================================================================
// chebwin
// ============================================================================

Array chebwin(int64_t M, double at, bool sym) {
  INS_CHECK(M >= 1, "chebwin: M must be >= 1");
  if (M == 1) {
    return full({1}, 1.0, DType::F64);
  }

  double atten = std::abs(at);
  int64_t M_work = sym ? M : M + 1;
  int64_t N = M_work - 1;
  double B = std::pow(10.0, atten / 20.0);
  double beta = std::cosh(std::acosh(B) / N);

  // Compute Chebyshev window directly in time domain
  // w[n] = sum_{k=0}^{N} T_N(beta*cos(pi*k/N)) * cos(2*pi*k*n/N) / N
  // with proper normalization
  std::vector<double> vals(M_work);

  for (int64_t n = 0; n < M_work; ++n) {
    double sum = 0.0;
    for (int64_t k = 0; k < M_work; ++k) {
      double x = beta * std::cos(M_PI * k / N);
      double Tn;
      double ax = std::abs(x);
      if (ax <= 1.0) {
        Tn = std::cos(N * std::acos(x));
      } else {
        // T_N(x) = (-1)^N * cosh(N * acosh(|x|)) for |x| > 1, x < 0
        // T_N(x) = cosh(N * acosh(x)) for x > 1
        double val = std::cosh(N * std::acosh(ax));
        Tn = (x < 0 && N % 2 == 1) ? -val : val;
      }
      sum += Tn * std::cos(2.0 * M_PI * k * n / M_work);
    }
    vals[n] = sum;
  }

  // Rearrange (fftshift)
  int64_t half = M_work / 2;
  std::vector<double> shifted(M_work);
  for (int64_t i = 0; i < M_work; ++i) {
    shifted[i] = vals[(i + half) % M_work];
  }

  // Fold to make symmetric
  for (int64_t i = 0; i < M_work / 2; ++i) {
    double avg =
        0.5 * (std::abs(shifted[i]) + std::abs(shifted[M_work - 1 - i]));
    shifted[i] = avg;
    shifted[M_work - 1 - i] = avg;
  }

  // Normalize so max = 1
  double max_val = 0.0;
  for (int64_t i = 0; i < M_work; ++i) {
    shifted[i] = std::abs(shifted[i]);
    if (shifted[i] > max_val)
      max_val = shifted[i];
  }
  if (max_val > 0) {
    for (int64_t i = 0; i < M_work; ++i) {
      shifted[i] /= max_val;
    }
  }

  if (!sym && M_work > M) {
    shifted.resize(M);
  }
  return to_array(shifted);
}

// ============================================================================
// taylor
// ============================================================================

Array taylor(int64_t M, int64_t nbar, double sll, bool norm, bool sym) {
  INS_CHECK(M >= 1, "taylor: M must be >= 1");
  INS_CHECK(nbar >= 1, "taylor: nbar must be >= 1");
  if (M == 1) {
    return full({1}, 1.0, DType::F64);
  }

  double B = std::pow(10.0, -sll / 20.0);
  double A = std::acosh(B) / M_PI;
  double A2 = A * A;
  double nbar_d = static_cast<double>(nbar);
  double s2 = nbar_d * nbar_d / (A2 + (nbar_d - 0.5) * (nbar_d - 0.5));

  std::vector<double> Fm(nbar, 0.0);
  for (int64_t m = 1; m < nbar; ++m) {
    double md = static_cast<double>(m);
    double num = 1.0;
    double den = 1.0;
    for (int64_t j = 1; j < nbar; ++j) {
      double jd = static_cast<double>(j);
      num *= (1.0 - s2 * md * md / (A2 + (jd - 0.5) * (jd - 0.5)));
      if (j != m) {
        den *= (1.0 - md * md / (jd * jd));
      }
    }
    Fm[m] = num / den;
  }

  // Compute on CPU (Taylor coefficients require nested loops), then transfer
  return apply_sym(M, sym, [&](int64_t N) {
    std::vector<double> vals(N);
    double N_d = static_cast<double>(N);
    for (int64_t i = 0; i < N; ++i) {
      double xi = i - N_d / 2.0 + 0.5;
      double val = 1.0;
      for (int64_t m = 1; m < nbar; ++m) {
        val += Fm[m] * std::cos(2.0 * M_PI * xi * m / N_d);
      }
      vals[i] = val;
    }
    if (norm) {
      double max_val = 0.0;
      for (int64_t i = 0; i < N; ++i)
        if (vals[i] > max_val)
          max_val = vals[i];
      if (max_val > 0) {
        for (int64_t i = 0; i < N; ++i)
          vals[i] /= max_val;
      }
    }
    return to_array(vals);
  });
}

// ============================================================================
// get_window
// ============================================================================

Array get_window(const std::string &window, int64_t Nx, bool fftbins) {
  bool sym = !fftbins;

  if (window == "boxcar" || window == "rect" || window == "ones" ||
      window == "dirichlet")
    return boxcar(Nx, sym);
  if (window == "triang")
    return triang(Nx, sym);
  if (window == "parzen")
    return parzen(Nx, sym);
  if (window == "bohman")
    return bohman(Nx, sym);
  if (window == "bartlett")
    return bartlett(Nx, sym);
  if (window == "cosine")
    return cosine(Nx, sym);
  if (window == "exponential")
    return exponential(Nx, -1.0, 1.0, sym);
  if (window == "blackman")
    return blackman(Nx, sym);
  if (window == "nuttall")
    return nuttall(Nx, sym);
  if (window == "blackmanharris")
    return blackmanharris(Nx, sym);
  if (window == "flattop")
    return flattop(Nx, sym);
  if (window == "hann" || window == "hanning")
    return hann(Nx, sym);
  if (window == "hamming")
    return hamming(Nx, sym);
  if (window == "tukey")
    return tukey(Nx, 0.5, sym);
  if (window == "barthann")
    return barthann(Nx, sym);
  if (window == "kaiser")
    return kaiser(Nx, 0.0, sym);
  if (window == "gaussian" || window == "gauss")
    return gaussian(Nx, 1.0, sym);
  if (window == "chebwin")
    return chebwin(Nx, 100.0, sym);
  if (window == "taylor")
    return taylor(Nx, 4, -30.0, true, sym);

  INS_CHECK(false, "get_window: unknown window name '", window, "'");
  return Array();
}

Array get_window(const std::string &window, double param, int64_t Nx,
                 bool fftbins) {
  bool sym = !fftbins;

  if (window == "kaiser")
    return kaiser(Nx, param, sym);
  if (window == "gaussian" || window == "gauss")
    return gaussian(Nx, param, sym);
  if (window == "general_gaussian")
    return general_gaussian(Nx, param, 1.0, sym);
  if (window == "chebwin")
    return chebwin(Nx, param, sym);
  if (window == "general_hamming")
    return general_hamming(Nx, param, sym);
  if (window == "tukey")
    return tukey(Nx, param, sym);
  if (window == "exponential")
    return exponential(Nx, -1.0, param, sym);
  if (window == "taylor")
    return taylor(Nx, static_cast<int64_t>(param), -30.0, true, sym);

  return get_window(window, Nx, fftbins);
}

// ============================================================================
// qmf — Quadrature Mirror Filter
// ============================================================================

std::pair<Array, Array> qmf(const Array &h_low) {
  INS_CHECK(h_low.defined(), "qmf: h_low is undefined");
  INS_CHECK(h_low.shape().ndim() == 1, "qmf: h_low must be 1D");

  Place cpu = CPUPlace();
  int64_t n = h_low.numel();
  Array h_cpu = h_low.to(cpu);

  // Highpass filter: h_high[k] = (-1)^k * h_low[k]
  // This is equivalent to modulating by exp(j*pi*k) = (-1)^k
  std::vector<double> mod(n);
  for (int64_t k = 0; k < n; ++k) {
    mod[k] = (k % 2 == 0) ? 1.0 : -1.0;
  }
  Array mod_arr = to_array(mod, h_low.dtype(), cpu);

  Array h_high = mul(h_cpu, mod_arr);

  // Move back to original device
  if (h_low.place().kind() != DeviceKind::CPU) {
    h_high = h_high.to(h_low.place());
  }

  return {h_low, h_high};
}

} // namespace signal
} // namespace ins
