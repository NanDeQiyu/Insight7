// demos/linalg_demo.cpp
// Demonstrates: linear algebra on CPU and GPU, multi-dtype, device check
#include "insight/insight.h"
#include "insight/ops/linalg.h"
#include <cmath>
#include <cstdio>

using namespace ins;

static void separator(const char *title) {
  printf("\n========================================\n");
  printf("  %s\n", title);
  printf("========================================\n");
}

static bool gpu_available() {
  try {
    set_device(GPUPlace(0));
    return true;
  } catch (...) {
    return false;
  }
}

static void run_cpu_linalg() {
  separator("CPU Linear Algebra");

  // MatMul F64
  {
    Array A =
        to_array({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), DType::F64, CPUPlace());
    Array B =
        to_array({5.0, 6.0, 7.0, 8.0}, Shape({2, 2}), DType::F64, CPUPlace());
    Array C = matmul(A, B);
    printf("MatMul F64:\n%s\n", to_string(C).c_str());
  }

  // MatMul F32
  {
    Array A = to_array({1.0f, 2.0f, 3.0f, 4.0f}, Shape({2, 2}), DType::F32,
                       CPUPlace());
    Array B = to_array({5.0f, 6.0f, 7.0f, 8.0f}, Shape({2, 2}), DType::F32,
                       CPUPlace());
    Array C = matmul(A, B);
    printf("MatMul F32:\n%s\n", to_string(C).c_str());
  }

  // Determinant
  {
    Array A =
        to_array({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), DType::F64, CPUPlace());
    printf("det([[1,2],[3,4]]) = %g\n", det(A).item<double>());
  }

  // Inverse
  {
    Array A =
        to_array({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), DType::F64, CPUPlace());
    Array A_inv = inv(A);
    printf("inv([[1,2],[3,4]]):\n%s\n", to_string(A_inv).c_str());

    Array I = matmul(A, A_inv);
    printf("A * A_inv (should be identity):\n%s\n", to_string(I).c_str());
  }

  // SVD
  {
    Array A = to_array({1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0},
                       Shape({3, 3}), DType::F64, CPUPlace());
    auto [U, S, VT] = svd(A, false);
    printf("SVD singular values: %s\n", to_string(S).c_str());
  }

  // Solve linear system
  {
    Array A = to_array({3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0},
                       Shape({3, 3}), DType::F64, CPUPlace());
    Array b = to_array({1.0, -2.0, 0.0}, Shape({3}), DType::F64, CPUPlace());
    Array x = solve(A, b);
    printf("Ax=b solution: %s\n", to_string(x).c_str());
  }
}

static void run_gpu_linalg() {
  separator("GPU Linear Algebra");

  // MatMul F64 on GPU
  {
    Array A =
        to_array({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), DType::F64, CPUPlace())
            .to(GPUPlace(0));
    Array B =
        to_array({5.0, 6.0, 7.0, 8.0}, Shape({2, 2}), DType::F64, CPUPlace())
            .to(GPUPlace(0));
    Array C = matmul(A, B);
    Array cpu_C = C.to(CPUPlace());
    printf("GPU MatMul F64:\n%s\n", to_string(cpu_C).c_str());
  }

  // MatMul F32 on GPU
  {
    Array A = to_array({1.0f, 2.0f, 3.0f, 4.0f}, Shape({2, 2}), DType::F32,
                       CPUPlace())
                  .to(GPUPlace(0));
    Array B = to_array({5.0f, 6.0f, 7.0f, 8.0f}, Shape({2, 2}), DType::F32,
                       CPUPlace())
                  .to(GPUPlace(0));
    Array C = matmul(A, B);
    Array cpu_C = C.to(CPUPlace());
    printf("GPU MatMul F32:\n%s\n", to_string(cpu_C).c_str());
  }

  // GPU det + inv
  {
    Array A =
        to_array({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), DType::F64, CPUPlace())
            .to(GPUPlace(0));
    printf("GPU det = %g\n", det(A).to(CPUPlace()).item<double>());

    Array A_inv = inv(A).to(CPUPlace());
    printf("GPU inv:\n%s\n", to_string(A_inv).c_str());
  }

  // GPU SVD (F32)
  {
    Array A = to_array({1.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f},
                       Shape({3, 3}), DType::F32, CPUPlace())
                  .to(GPUPlace(0));
    auto [U, S, VT] = svd(A, false);
    Array cpu_S = S.to(CPUPlace());
    printf("GPU SVD singular values (F32): %s\n", to_string(cpu_S).c_str());
  }
}

int main() {
  ins::init({"cpu", "cuda"});

  printf("Insight7 Linear Algebra Demo\n");
  printf("OpenBLAS: %s\n", is_compiled_with_openblas() ? "yes" : "no");

  run_cpu_linalg();

  if (gpu_available()) {
    run_gpu_linalg();
  } else {
    printf("\n[GPU not available, skipping GPU linalg demo]\n");
  }

  printf("\nDone!\n");
  return 0;
}
