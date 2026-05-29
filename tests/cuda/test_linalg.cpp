// tests/cuda/test_linalg.cpp
/**
 * @file test_linalg.cpp
 * @brief CUDA linalg tests — mirrors CPU tests with GPU dispatch.
 *
 * Operations with native CUDA kernels run on GPU directly.
 * Operations returning C_FALLBACK are automatically transferred to CPU
 * by the framework's fallback mechanism.
 */

#include "insight/insight.h"
#include "insight/ops/linalg.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace ins;

class LinalgTestGPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu", "cuda"});
    try {
      set_device(GPUPlace(0));
    } catch (...) {
      GTEST_SKIP() << "GPU not available";
    }
    if (!ins::is_compiled_with_openblas()) {
      GTEST_SKIP() << "Skipping linalg GPU tests: OpenBLAS not available "
                      "(needed for CPU fallback)";
    }
  }
};

// ============================================================================
// Helper functions
// ============================================================================

static Array make_gpu_matrix_f64(int rows, int cols,
                                 const std::vector<double> &data) {
  Array result(Shape({rows, cols}), DType::F64, CPUPlace());
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result.to(GPUPlace(0));
}

static Array make_gpu_matrix_f32(int rows, int cols,
                                 const std::vector<float> &data) {
  Array result(Shape({rows, cols}), DType::F32, CPUPlace());
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result.to(GPUPlace(0));
}

static Array make_gpu_vector_f64(const std::vector<double> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F64,
               CPUPlace());
  std::memcpy(result.data<double>(), data.data(), data.size() * sizeof(double));
  return result.to(GPUPlace(0));
}

static Array make_gpu_vector_f32(const std::vector<float> &data) {
  Array result(Shape({static_cast<int64_t>(data.size())}), DType::F32,
               CPUPlace());
  std::memcpy(result.data<float>(), data.data(), data.size() * sizeof(float));
  return result.to(GPUPlace(0));
}

static bool approx_equal(double a, double b, double rtol = 1e-5,
                         double atol = 1e-7) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool approx_equal(float a, float b, float rtol = 1e-4f,
                         float atol = 1e-5f) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

static bool check_matrix_equal(const Array &gpu_A, const Array &gpu_B,
                               double rtol = 1e-4) {
  Array A = gpu_A.to(CPUPlace());
  Array B = gpu_B.to(CPUPlace());
  if (A.shape() != B.shape())
    return false;
  if (A.dtype() != B.dtype())
    return false;
  int64_t n = A.numel();
  if (A.dtype() == DType::F64) {
    const double *a = A.data<double>();
    const double *b = B.data<double>();
    for (int64_t i = 0; i < n; ++i) {
      if (!approx_equal(a[i], b[i], rtol))
        return false;
    }
  } else if (A.dtype() == DType::F32) {
    const float *a = A.data<float>();
    const float *b = B.data<float>();
    for (int64_t i = 0; i < n; ++i) {
      if (!approx_equal(a[i], b[i], static_cast<float>(rtol)))
        return false;
    }
  } else {
    return false;
  }
  return true;
}

// ============================================================================
// matmul tests
// ============================================================================

TEST_F(LinalgTestGPU, MatMul2x2F64) {
  Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
  Array B = make_gpu_matrix_f64(2, 2, {5, 6, 7, 8});
  Array C = matmul(A, B);
  Array cpu_C = C.to(CPUPlace());
  const double *c = cpu_C.data<double>();
  EXPECT_NEAR(c[0], 19, 1e-5);
  EXPECT_NEAR(c[1], 22, 1e-5);
  EXPECT_NEAR(c[2], 43, 1e-5);
  EXPECT_NEAR(c[3], 50, 1e-5);
}

TEST_F(LinalgTestGPU, MatMul2x2F32) {
  Array A = make_gpu_matrix_f32(2, 2, {1.0f, 2.0f, 3.0f, 4.0f});
  Array B = make_gpu_matrix_f32(2, 2, {5.0f, 6.0f, 7.0f, 8.0f});
  Array C = matmul(A, B);
  Array cpu_C = C.to(CPUPlace());
  const float *c = cpu_C.data<float>();
  EXPECT_NEAR(c[0], 19.0f, 1e-4f);
  EXPECT_NEAR(c[1], 22.0f, 1e-4f);
  EXPECT_NEAR(c[2], 43.0f, 1e-4f);
  EXPECT_NEAR(c[3], 50.0f, 1e-4f);
}

TEST_F(LinalgTestGPU, MatMulNonSquare) {
  {
    Array A = make_gpu_matrix_f64(2, 3, {1, 2, 3, 4, 5, 6});
    Array B = make_gpu_matrix_f64(3, 2, {7, 8, 9, 10, 11, 12});
    Array C = matmul(A, B);
    Array cpu_C = C.to(CPUPlace());
    const double *c = cpu_C.data<double>();
    EXPECT_NEAR(c[0], 58, 1e-5);
    EXPECT_NEAR(c[1], 64, 1e-5);
    EXPECT_NEAR(c[2], 139, 1e-5);
    EXPECT_NEAR(c[3], 154, 1e-5);
  }
  {
    Array A = make_gpu_matrix_f32(2, 3, {1, 2, 3, 4, 5, 6});
    Array B = make_gpu_matrix_f32(3, 2, {7, 8, 9, 10, 11, 12});
    Array C = matmul(A, B);
    Array cpu_C = C.to(CPUPlace());
    const float *c = cpu_C.data<float>();
    EXPECT_NEAR(c[0], 58.0f, 1e-4f);
    EXPECT_NEAR(c[1], 64.0f, 1e-4f);
    EXPECT_NEAR(c[2], 139.0f, 1e-4f);
    EXPECT_NEAR(c[3], 154.0f, 1e-4f);
  }
}

// ============================================================================
// det tests
// ============================================================================

TEST_F(LinalgTestGPU, Det2x2) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<double>(), -2.0, 1e-5);
  }
  {
    Array A = make_gpu_matrix_f32(2, 2, {1, 2, 3, 4});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), -2.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Det3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<double>(), -1.0, 1e-5);
  }
  {
    Array A = make_gpu_matrix_f32(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), -1.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, DetIdentity) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<double>(), 1.0, 1e-5);
  }
  {
    Array A = make_gpu_matrix_f32(3, 3, {1, 0, 0, 0, 1, 0, 0, 0, 1});
    Array d = det(A);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), 1.0f, 1e-4f);
  }
}

// ============================================================================
// slogdet tests
// ============================================================================

TEST_F(LinalgTestGPU, Slogdet2x2) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    auto [sign, logdet] = slogdet(A);
    Array cpu_sign = sign.to(CPUPlace());
    Array cpu_logdet = logdet.to(CPUPlace());
    EXPECT_NEAR(cpu_sign.item<double>(), -1.0, 1e-5);
    EXPECT_NEAR(cpu_logdet.item<double>(), std::log(2.0), 1e-5);
  }
  {
    Array A = make_gpu_matrix_f32(2, 2, {1, 2, 3, 4});
    auto [sign, logdet] = slogdet(A);
    Array cpu_sign = sign.to(CPUPlace());
    Array cpu_logdet = logdet.to(CPUPlace());
    EXPECT_NEAR(cpu_sign.item<float>(), -1.0f, 1e-4f);
    EXPECT_NEAR(cpu_logdet.item<double>(), std::log(2.0), 1e-5);
  }
}

// ============================================================================
// inv tests
// ============================================================================

TEST_F(LinalgTestGPU, Inv2x2) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array invA = inv(A);
    Array cpu_invA = invA.to(CPUPlace());
    const double *data = cpu_invA.data<double>();
    EXPECT_NEAR(data[0], -2.0, 1e-5);
    EXPECT_NEAR(data[1], 1.0, 1e-5);
    EXPECT_NEAR(data[2], 1.5, 1e-5);
    EXPECT_NEAR(data[3], -0.5, 1e-5);

    // Verify A * invA = I
    Array I = matmul(A, invA);
    Array cpu_I = I.to(CPUPlace());
    const double *i = cpu_I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-4);
    EXPECT_NEAR(i[1], 0.0, 1e-4);
    EXPECT_NEAR(i[2], 0.0, 1e-4);
    EXPECT_NEAR(i[3], 1.0, 1e-4);
  }
  {
    Array A = make_gpu_matrix_f32(2, 2, {1, 2, 3, 4});
    Array invA = inv(A);
    Array cpu_invA = invA.to(CPUPlace());
    const float *data = cpu_invA.data<float>();
    EXPECT_NEAR(data[0], -2.0f, 1e-4f);
    EXPECT_NEAR(data[1], 1.0f, 1e-4f);
    EXPECT_NEAR(data[2], 1.5f, 1e-4f);
    EXPECT_NEAR(data[3], -0.5f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Inv3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    Array invA = inv(A);
    Array I = matmul(A, invA);
    Array cpu_I = I.to(CPUPlace());
    const double *i = cpu_I.data<double>();
    EXPECT_NEAR(i[0], 1.0, 1e-4);
    EXPECT_NEAR(i[1], 0.0, 1e-4);
    EXPECT_NEAR(i[2], 0.0, 1e-4);
    EXPECT_NEAR(i[3], 0.0, 1e-4);
    EXPECT_NEAR(i[4], 1.0, 1e-4);
    EXPECT_NEAR(i[5], 0.0, 1e-4);
    EXPECT_NEAR(i[6], 0.0, 1e-4);
    EXPECT_NEAR(i[7], 0.0, 1e-4);
    EXPECT_NEAR(i[8], 1.0, 1e-4);
  }
}

// ============================================================================
// solve tests
// ============================================================================

TEST_F(LinalgTestGPU, Solve3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array b = make_gpu_vector_f64({1.0, -2.0, 0.0});
    Array x = solve(A, b);
    Array cpu_x = x.to(CPUPlace());
    const double *data = cpu_x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-8);
    EXPECT_NEAR(data[1], -2.0, 1e-8);
    EXPECT_NEAR(data[2], -2.0, 1e-8);
  }
  {
    Array A = make_gpu_matrix_f32(3, 3, {3, 2, -1, 2, -2, 4, -1, 0.5, -1});
    Array b = make_gpu_vector_f32({1.0f, -2.0f, 0.0f});
    Array x = solve(A, b);
    Array cpu_x = x.to(CPUPlace());
    const float *data = cpu_x.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-4f);
    EXPECT_NEAR(data[1], -2.0f, 1e-4f);
    EXPECT_NEAR(data[2], -2.0f, 1e-4f);
  }
}

// ============================================================================
// cholesky tests
// ============================================================================

TEST_F(LinalgTestGPU, Cholesky3x3) {
  {
    Array A =
        make_gpu_matrix_f64(3, 3, {4, 12, -16, 12, 37, -43, -16, -43, 98});
    Array L = cholesky(A, true);
    Array LLT = matmul(L, transpose(L));
    EXPECT_TRUE(check_matrix_equal(A, LLT, 1e-4));
  }
}

// ============================================================================
// qr tests
// ============================================================================

TEST_F(LinalgTestGPU, QR3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {12, -51, 4, 6, 167, -68, -4, 24, -41});
    auto [Q, R] = qr(A, "reduced");
    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-4));

    // Check Q is orthogonal
    Array QTQ = matmul(transpose(Q), Q);
    Array cpu_QTQ = QTQ.to(CPUPlace());
    const double *qtq_data = cpu_QTQ.data<double>();
    EXPECT_NEAR(qtq_data[0], 1.0, 1e-4);
    EXPECT_NEAR(qtq_data[4], 1.0, 1e-4);
    EXPECT_NEAR(qtq_data[8], 1.0, 1e-4);
  }
}

TEST_F(LinalgTestGPU, QR3x2) {
  {
    Array A = make_gpu_matrix_f64(3, 2, {1, 2, 3, 4, 5, 6});
    auto [Q, R] = qr(A, "reduced");
    Array cpu_Q = Q.to(CPUPlace());
    EXPECT_EQ(cpu_Q.shape(), Shape({3, 2}));
    Array QR = matmul(Q, R);
    EXPECT_TRUE(check_matrix_equal(A, QR, 1e-4));
  }
}

// ============================================================================
// svd tests
// ============================================================================

TEST_F(LinalgTestGPU, SVD3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    auto [U, S, VT] = svd(A, false);
    Array cpu_S = S.to(CPUPlace());
    const double *s = cpu_S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-4);
    EXPECT_NEAR(s[1], 2.0, 1e-4);
    EXPECT_NEAR(s[2], 1.0, 1e-4);
  }
  {
    Array A = make_gpu_matrix_f32(
        3, 3, {1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    auto [U, S, VT] = svd(A, false);
    Array cpu_S = S.to(CPUPlace());
    const float *s = cpu_S.data<float>();
    EXPECT_NEAR(s[0], 3.0f, 1e-3f);
    EXPECT_NEAR(s[1], 2.0f, 1e-3f);
    EXPECT_NEAR(s[2], 1.0f, 1e-3f);
  }
}

TEST_F(LinalgTestGPU, SVDvals) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 0, 0, 0, 2, 0, 0, 0, 3});
    Array S = svdvals(A);
    Array cpu_S = S.to(CPUPlace());
    const double *s = cpu_S.data<double>();
    EXPECT_NEAR(s[0], 3.0, 1e-4);
    EXPECT_NEAR(s[1], 2.0, 1e-4);
    EXPECT_NEAR(s[2], 1.0, 1e-4);
  }
}

// ============================================================================
// eigh / eigvalsh tests
// ============================================================================

TEST_F(LinalgTestGPU, Eigh3x3) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    auto [vals, vecs] = eigh(A, "L");
    Array cpu_vals = vals.to(CPUPlace());
    const double *v = cpu_vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-2);
    EXPECT_NEAR(v[1], 2.0, 1e-2);
    EXPECT_NEAR(v[2], 3.4142, 1e-2);
  }
}

TEST_F(LinalgTestGPU, Eigvalsh) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    Array vals = eigvalsh(A, "L");
    Array cpu_vals = vals.to(CPUPlace());
    const double *v = cpu_vals.data<double>();
    EXPECT_NEAR(v[0], 0.5858, 1e-2);
    EXPECT_NEAR(v[1], 2.0, 1e-2);
    EXPECT_NEAR(v[2], 3.4142, 1e-2);
  }
}

// ============================================================================
// dot / outer tests
// ============================================================================

TEST_F(LinalgTestGPU, Dot) {
  {
    Array a = make_gpu_vector_f64({1.0, 2.0, 3.0});
    Array b = make_gpu_vector_f64({4.0, 5.0, 6.0});
    Array d = dot(a, b);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<double>(), 32.0, 1e-5);
  }
  {
    Array a = make_gpu_vector_f32({1.0f, 2.0f, 3.0f});
    Array b = make_gpu_vector_f32({4.0f, 5.0f, 6.0f});
    Array d = dot(a, b);
    Array cpu_d = d.to(CPUPlace());
    EXPECT_NEAR(cpu_d.item<float>(), 32.0f, 1e-4f);
  }
}

TEST_F(LinalgTestGPU, Outer) {
  {
    Array a = make_gpu_vector_f64({1.0, 2.0, 3.0});
    Array b = make_gpu_vector_f64({4.0, 5.0, 6.0});
    Array o = outer(a, b);
    Array cpu_o = o.to(CPUPlace());
    const double *data = cpu_o.data<double>();
    EXPECT_NEAR(data[0], 4.0, 1e-5);
    EXPECT_NEAR(data[4], 10.0, 1e-5);
    EXPECT_NEAR(data[8], 18.0, 1e-5);
  }
}

// ============================================================================
// matrix_power tests
// ============================================================================

TEST_F(LinalgTestGPU, MatrixPower2x2) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array A2 = matrix_power(A, 2);
    Array cpu_A2 = A2.to(CPUPlace());
    const double *data = cpu_A2.data<double>();
    EXPECT_NEAR(data[0], 7.0, 1e-5);
    EXPECT_NEAR(data[1], 10.0, 1e-5);
    EXPECT_NEAR(data[2], 15.0, 1e-5);
    EXPECT_NEAR(data[3], 22.0, 1e-5);
  }
}

TEST_F(LinalgTestGPU, MatrixPowerZero) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array A0 = matrix_power(A, 0);
    Array cpu_A0 = A0.to(CPUPlace());
    const double *data = cpu_A0.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-5);
    EXPECT_NEAR(data[1], 0.0, 1e-5);
    EXPECT_NEAR(data[2], 0.0, 1e-5);
    EXPECT_NEAR(data[3], 1.0, 1e-5);
  }
}

// ============================================================================
// trace tests
// ============================================================================

TEST_F(LinalgTestGPU, Trace) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Array t = trace(A);
    Array cpu_t = t.to(CPUPlace());
    EXPECT_NEAR(cpu_t.item<double>(), 15.0, 1e-5);
  }
}

// ============================================================================
// norm tests
// ============================================================================

TEST_F(LinalgTestGPU, NormVector2) {
  {
    Array v = make_gpu_vector_f64({3.0, 4.0});
    Array n = norm(v);
    Array cpu_n = n.to(CPUPlace());
    EXPECT_NEAR(cpu_n.item<double>(), 5.0, 1e-5);
  }
}

TEST_F(LinalgTestGPU, NormMatrixFrobenius) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array n = norm(A, 2);
    Array cpu_n = n.to(CPUPlace());
    double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0);
    EXPECT_NEAR(cpu_n.item<double>(), expected, 1e-5);
  }
}

// ============================================================================
// solve_triangular tests
// ============================================================================

TEST_F(LinalgTestGPU, SolveTriangularUpper) {
  {
    Array U = make_gpu_matrix_f64(3, 3, {1, 2, 3, 0, 4, 5, 0, 0, 6});
    Array b = make_gpu_vector_f64({1.0, 2.0, 3.0});
    Array x = solve_triangular(U, b, false, false);
    Array cpu_x = x.to(CPUPlace());
    const double *data = cpu_x.data<double>();
    EXPECT_NEAR(data[0], -0.25, 1e-4);
    EXPECT_NEAR(data[1], -0.125, 1e-4);
    EXPECT_NEAR(data[2], 0.5, 1e-4);
  }
}

TEST_F(LinalgTestGPU, SolveTriangularLower) {
  {
    Array L = make_gpu_matrix_f64(3, 3, {2, 0, 0, 1, 2, 0, 1, 1, 2});
    Array b = make_gpu_vector_f64({4.0, 5.0, 6.0});
    Array x = solve_triangular(L, b, true, false);
    Array cpu_x = x.to(CPUPlace());
    const double *data = cpu_x.data<double>();
    EXPECT_NEAR(data[0], 2.0, 1e-4);
    EXPECT_NEAR(data[1], 1.5, 1e-4);
    EXPECT_NEAR(data[2], 1.25, 1e-4);
  }
}

// ============================================================================
// lu tests
// ============================================================================

TEST_F(LinalgTestGPU, LuDecomposition) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    auto [LU, pivots] = lu(A, true);
    auto [P, L, U] = lu_unpack(LU, pivots);
    Array PA = matmul(P, A);
    Array LU_mat = matmul(L, U);
    EXPECT_TRUE(check_matrix_equal(PA, LU_mat, 1e-4));
  }
}

// ============================================================================
// lq tests
// ============================================================================

TEST_F(LinalgTestGPU, LqDecomposition) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 2, 5, 3, 1, 0, 8});
    auto [L, Q] = lq(A, "reduced");
    Array LQ = matmul(L, Q);
    EXPECT_TRUE(check_matrix_equal(A, LQ, 1e-4));
  }
}

// ============================================================================
// Fallback tests (operations that return C_FALLBACK)
// These verify the fallback mechanism works end-to-end.
// ============================================================================

TEST_F(LinalgTestGPU, CondFallback) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array c = cond(A, 1);
    Array cpu_c = c.to(CPUPlace());
    EXPECT_NEAR(cpu_c.item<double>(), 21.0, 1.0);
  }
}

TEST_F(LinalgTestGPU, MatrixRankFallback) {
  {
    Array A = make_gpu_matrix_f64(3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    Array r = matrix_rank(A);
    Array cpu_r = r.to(CPUPlace());
    EXPECT_EQ(cpu_r.item<int64_t>(), 2);
  }
}

TEST_F(LinalgTestGPU, PinvFallback) {
  {
    Array A = make_gpu_matrix_f64(2, 2, {1, 2, 3, 4});
    Array pinvA = pinv(A);
    Array I = matmul(A, pinvA);
    Array cpu_I = I.to(CPUPlace());
    EXPECT_NEAR(cpu_I.data<double>()[0], 1.0, 1e-4);
    EXPECT_NEAR(cpu_I.data<double>()[3], 1.0, 1e-4);
  }
}

TEST_F(LinalgTestGPU, LstsqFallback) {
  {
    Array A = make_gpu_matrix_f64(3, 2, {1, 1, 1, 2, 1, 3});
    Array b = make_gpu_vector_f64({2.0, 3.0, 4.0});
    Array x = lstsq(A, b);
    Array cpu_x = x.to(CPUPlace());
    const double *data = cpu_x.data<double>();
    EXPECT_NEAR(data[0], 1.0, 1e-4);
    EXPECT_NEAR(data[1], 1.0, 1e-4);
  }
}

TEST_F(LinalgTestGPU, CovFallback) {
  std::vector<double> data = {1.0, 2.0, 3.0, 4.0,  5.0,  6.0,
                              7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  Array x(Shape({3, 4}), DType::F64, CPUPlace());
  std::memcpy(x.data<double>(), data.data(), data.size() * sizeof(double));
  Array gpu_x = x.to(GPUPlace(0));
  Array c = ins::cov(gpu_x);
  Array cpu_c = c.to(CPUPlace());
  EXPECT_EQ(cpu_c.shape(), Shape({3, 3}));
  const double *c_data = cpu_c.data<double>();
  EXPECT_NEAR(c_data[0], 5.0 / 3.0, 1e-5);
  EXPECT_NEAR(c_data[1], 5.0 / 3.0, 1e-5);
}
