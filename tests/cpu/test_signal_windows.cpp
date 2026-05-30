// tests/cpu/test_signal_windows.cpp
#include "insight/insight.h"
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ins;

class SignalWindowsTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
    set_device(CPUPlace());
  }
};

namespace {
template <typename T>
void expect_float_equal(const Array &arr, const std::vector<T> &expected,
                        T tol = 1e-6) {
  ASSERT_EQ(arr.numel(), static_cast<int64_t>(expected.size()));
  const T *data = arr.data<T>();
  for (int64_t i = 0; i < arr.numel(); ++i) {
    EXPECT_NEAR(data[i], expected[i], tol) << "mismatch at index " << i;
  }
}
} // namespace

// ========== boxcar ==========

TEST_F(SignalWindowsTestCPU, BoxcarBasic) {
  Array w = signal::boxcar(5);
  EXPECT_EQ(w.numel(), 5);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(w.data<double>()[i], 1.0);
  }
}

TEST_F(SignalWindowsTestCPU, BoxcarSize1) {
  Array w = signal::boxcar(1);
  EXPECT_EQ(w.numel(), 1);
  EXPECT_DOUBLE_EQ(w.data<double>()[0], 1.0);
}

TEST_F(SignalWindowsTestCPU, BoxcarPeriodic) {
  Array w = signal::boxcar(5, false);
  EXPECT_EQ(w.numel(), 5);
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_DOUBLE_EQ(w.data<double>()[i], 1.0);
  }
}

// ========== triang ==========

TEST_F(SignalWindowsTestCPU, TriangOdd) {
  // scipy.signal.windows.triang(5): [0.333, 0.667, 1.0, 0.667, 0.333]
  Array w = signal::triang(5);
  EXPECT_EQ(w.numel(), 5);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], 1.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[1], 2.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[3], 2.0 / 3.0, 1e-10);
  EXPECT_NEAR(d[4], 1.0 / 3.0, 1e-10);
}

TEST_F(SignalWindowsTestCPU, TriangEven) {
  // scipy.signal.windows.triang(4): [0.25, 0.75, 0.75, 0.25]
  Array w = signal::triang(4);
  EXPECT_EQ(w.numel(), 4);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], 0.25, 1e-10);
  EXPECT_NEAR(d[1], 0.75, 1e-10);
  EXPECT_NEAR(d[2], 0.75, 1e-10);
  EXPECT_NEAR(d[3], 0.25, 1e-10);
}

// ========== parzen ==========

TEST_F(SignalWindowsTestCPU, ParzenBasic) {
  // scipy.signal.windows.parzen(8) = [0.00390625, 0.10546875, 0.47265625,
  // 0.91796875, 0.91796875, 0.47265625, 0.10546875, 0.00390625]
  Array w = signal::parzen(8);
  EXPECT_EQ(w.numel(), 8);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], 0.00390625, 1e-6);
  EXPECT_NEAR(d[1], 0.10546875, 1e-6);
  EXPECT_NEAR(d[2], 0.47265625, 1e-6);
  EXPECT_NEAR(d[3], 0.91796875, 1e-6);
  // Symmetric
  EXPECT_NEAR(d[4], d[3], 1e-10);
  EXPECT_NEAR(d[5], d[2], 1e-10);
  EXPECT_NEAR(d[6], d[1], 1e-10);
  EXPECT_NEAR(d[7], d[0], 1e-10);
}

// ========== bohman ==========

TEST_F(SignalWindowsTestCPU, BohmanEndpoints) {
  Array w = signal::bohman(11);
  const double *d = w.data<double>();
  // Endpoints should be 0
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[10], 0.0, 1e-10);
  // Center should be 1
  EXPECT_NEAR(d[5], 1.0, 1e-6);
}

// ========== bartlett ==========

TEST_F(SignalWindowsTestCPU, BartlettBasic) {
  // scipy.signal.windows.bartlett(5) = [0, 0.5, 1, 0.5, 0]
  Array w = signal::bartlett(5);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[3], 0.5, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
}

// ========== cosine ==========

TEST_F(SignalWindowsTestCPU, CosineBasic) {
  // scipy.signal.windows.cosine(4) = [sin(pi/8), sin(3pi/8), sin(5pi/8),
  // sin(7pi/8)]
  Array w = signal::cosine(4);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], std::sin(M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[1], std::sin(3.0 * M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[2], std::sin(5.0 * M_PI / 8.0), 1e-10);
  EXPECT_NEAR(d[3], std::sin(7.0 * M_PI / 8.0), 1e-10);
}

// ========== blackman ==========

TEST_F(SignalWindowsTestCPU, BlackmanBasic) {
  // scipy.signal.windows.blackman(5) uses DFT-even by default in scipy
  // but we test symmetric here
  Array w = signal::blackman(5);
  const double *d = w.data<double>();
  // Endpoints should be ~0
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
  // Center should be ~1
  EXPECT_NEAR(d[2], 1.0, 1e-6);
  // Symmetric
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== hann ==========

TEST_F(SignalWindowsTestCPU, HannBasic) {
  // scipy.signal.windows.hann(5, sym=True):
  // w[n] = 0.5 - 0.5*cos(2*pi*n/4) for n=0..4
  Array w = signal::hann(5);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[4], 0.0, 1e-10);
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  // d[1] = 0.5 - 0.5*cos(pi/2) = 0.5
  EXPECT_NEAR(d[1], 0.5, 1e-10);
  EXPECT_NEAR(d[3], 0.5, 1e-10);
}

// ========== hamming ==========

TEST_F(SignalWindowsTestCPU, HammingBasic) {
  Array w = signal::hamming(5);
  const double *d = w.data<double>();
  // Endpoints should be 0.08 (not 0!)
  EXPECT_NEAR(d[0], 0.08, 1e-10);
  EXPECT_NEAR(d[4], 0.08, 1e-10);
  // Center
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  // Symmetric
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== kaiser ==========

TEST_F(SignalWindowsTestCPU, KaiserBasic) {
  // Kaiser with beta=0 should be rectangular
  Array w_rect = signal::kaiser(5, 0.0);
  const double *d = w_rect.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], 1.0, 1e-10);
  }
}

TEST_F(SignalWindowsTestCPU, KaiserSymmetry) {
  Array w = signal::kaiser(11, 5.0);
  const double *d = w.data<double>();
  // Should be symmetric
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10);
  }
  // Center should be 1.0 (I0(0)/I0(beta) = 1)
  EXPECT_NEAR(d[5], 1.0, 1e-10);
  // Endpoints should be > 0
  EXPECT_GT(d[0], 0.0);
}

// ========== gaussian ==========

TEST_F(SignalWindowsTestCPU, GaussianBasic) {
  // Gaussian with std=1, M=5
  Array w = signal::gaussian(5, 1.0);
  const double *d = w.data<double>();
  // Center should be 1
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  // Symmetric
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  // d[0] = exp(-0.5 * (2/1)^2) = exp(-2) ≈ 0.1353
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== general_gaussian ==========

TEST_F(SignalWindowsTestCPU, GeneralGaussianBasic) {
  // p=1, sig=1 should be same as gaussian with std=1
  Array w = signal::general_gaussian(5, 1.0, 1.0);
  const double *d = w.data<double>();
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== general_cosine ==========

TEST_F(SignalWindowsTestCPU, GeneralCosineHFT90D) {
  // HFT90D coefficients — periodic window (sym=false)
  // This window legitimately has negative values (high-frequency flat-top)
  std::vector<double> HFT90D = {1, 1.942604, 1.340318, 0.440811, 0.043097};
  Array w = signal::general_cosine(64, HFT90D, false);
  EXPECT_EQ(w.numel(), 64);
  const double *d = w.data<double>();
  // Max should be near center and positive
  double max_val = d[0];
  for (int64_t i = 1; i < 64; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_GT(max_val, 4.0);
  // Endpoint should be near 0
  EXPECT_NEAR(d[0], 0.0, 1e-10);
  EXPECT_NEAR(d[63], 0.0, 1e-3);
}

// ========== tukey ==========

TEST_F(SignalWindowsTestCPU, TukeyAlpha0) {
  // alpha=0 should be rectangular
  Array w = signal::tukey(10, 0.0);
  const double *d = w.data<double>();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(d[i], 1.0, 1e-10);
  }
}

TEST_F(SignalWindowsTestCPU, TukeyAlpha1) {
  // alpha=1 should be Hann
  Array w_tukey = signal::tukey(10, 1.0);
  Array w_hann = signal::hann(10);
  const double *dt = w_tukey.data<double>();
  const double *dh = w_hann.data<double>();
  for (int64_t i = 0; i < 10; ++i) {
    EXPECT_NEAR(dt[i], dh[i], 1e-10);
  }
}

// ========== barthann ==========

TEST_F(SignalWindowsTestCPU, BarthannBasic) {
  Array w = signal::barthann(11);
  const double *d = w.data<double>();
  // Center should be close to 1
  EXPECT_NEAR(d[5], 1.0, 0.02);
  // Symmetric
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-10);
  }
}

// ========== exponential ==========

TEST_F(SignalWindowsTestCPU, ExponentialBasic) {
  Array w = signal::exponential(5, -1.0, 1.0);
  const double *d = w.data<double>();
  // Center should be 1 (center defaults to (M-1)/2 = 2)
  EXPECT_NEAR(d[2], 1.0, 1e-10);
  // Symmetric
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
  // d[0] = exp(-2/1) = exp(-2)
  EXPECT_NEAR(d[0], std::exp(-2.0), 1e-10);
}

// ========== chebwin ==========

TEST_F(SignalWindowsTestCPU, ChebwinBasic) {
  Array w = signal::chebwin(11, 50.0);
  const double *d = w.data<double>();
  EXPECT_EQ(w.numel(), 11);
  // Should be symmetric
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], d[10 - i], 1e-6);
  }
  // All values should be in [0, 1]
  for (int64_t i = 0; i < 11; ++i) {
    EXPECT_GE(d[i], 0.0);
    EXPECT_LE(d[i], 1.0 + 1e-6);
  }
  // Max should be 1
  double max_val = 0;
  for (int64_t i = 0; i < 11; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_NEAR(max_val, 1.0, 1e-6);
}

// ========== taylor ==========

TEST_F(SignalWindowsTestCPU, TaylorBasic) {
  Array w = signal::taylor(32, 4, -30.0, true);
  const double *d = w.data<double>();
  EXPECT_EQ(w.numel(), 32);
  // Should be symmetric
  for (int64_t i = 0; i < 16; ++i) {
    EXPECT_NEAR(d[i], d[31 - i], 1e-6);
  }
  // All values should be in [0, 1]
  for (int64_t i = 0; i < 32; ++i) {
    EXPECT_GE(d[i], 0.0);
    EXPECT_LE(d[i], 1.0 + 1e-6);
  }
  // Max should be 1 (normalized)
  double max_val = 0;
  for (int64_t i = 0; i < 32; ++i) {
    if (d[i] > max_val)
      max_val = d[i];
  }
  EXPECT_NEAR(max_val, 1.0, 1e-6);
}

// ========== get_window ==========

TEST_F(SignalWindowsTestCPU, GetWindowBoxcar) {
  Array w = signal::get_window("boxcar", 8);
  EXPECT_EQ(w.numel(), 8);
  for (int64_t i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(w.data<double>()[i], 1.0);
  }
}

TEST_F(SignalWindowsTestCPU, GetWindowHann) {
  // get_window("hann", 5) uses fftbins=true (periodic), compare with
  // hann(5, false)
  Array w = signal::get_window("hann", 5);
  Array ref = signal::hann(5, false);
  const double *d = w.data<double>();
  const double *r = ref.data<double>();
  for (int64_t i = 0; i < 5; ++i) {
    EXPECT_NEAR(d[i], r[i], 1e-10);
  }
}

TEST_F(SignalWindowsTestCPU, GetWindowKaiser) {
  // get_window("kaiser", 5.0, 11) uses fftbins=true (periodic), compare with
  // kaiser(11, 5.0, false)
  Array w = signal::get_window("kaiser", 5.0, 11);
  Array ref = signal::kaiser(11, 5.0, false);
  const double *d = w.data<double>();
  const double *r = ref.data<double>();
  for (int64_t i = 0; i < 11; ++i) {
    EXPECT_NEAR(d[i], r[i], 1e-10);
  }
}

// ========== nuttall / blackmanharris / flattop ==========

TEST_F(SignalWindowsTestCPU, NuttallBasic) {
  Array w = signal::nuttall(5);
  const double *d = w.data<double>();
  // Nuttall endpoints are very small but not exactly 0
  EXPECT_NEAR(d[0], 0.0003628, 1e-4);
  EXPECT_NEAR(d[4], 0.0003628, 1e-4);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWindowsTestCPU, BlackmanharrisBasic) {
  Array w = signal::blackmanharris(5);
  const double *d = w.data<double>();
  // Blackman-Harris endpoints are very small but not exactly 0
  EXPECT_NEAR(d[0], 6.0e-5, 1e-4);
  EXPECT_NEAR(d[4], 6.0e-5, 1e-4);
  EXPECT_NEAR(d[2], 1.0, 1e-6);
}

TEST_F(SignalWindowsTestCPU, FlattopBasic) {
  Array w = signal::flattop(5);
  const double *d = w.data<double>();
  EXPECT_EQ(w.numel(), 5);
  // Symmetric
  EXPECT_NEAR(d[0], d[4], 1e-10);
  EXPECT_NEAR(d[1], d[3], 1e-10);
}

// ========== periodic (sym=false) ==========

TEST_F(SignalWindowsTestCPU, HannPeriodic) {
  Array w_sym = signal::hann(10, true);
  Array w_per = signal::hann(10, false);
  // Both should have 10 elements
  EXPECT_EQ(w_sym.numel(), 10);
  EXPECT_EQ(w_per.numel(), 10);
  // Periodic should differ from symmetric
  const double *ds = w_sym.data<double>();
  const double *dp = w_per.data<double>();
  // Last sample: periodic should not equal symmetric
  // (periodic uses M+1=11 internally, truncated to 10)
  // Actually for hann, periodic last sample != 0
  // while symmetric last sample = 0
  EXPECT_NE(ds[9], dp[9]);
}
