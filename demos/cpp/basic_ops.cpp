// demos/basic_ops.cpp
// Demonstrates: array creation, arithmetic, multi-dtype, broadcasting
#include "insight/insight.h"
#include <cstdio>

using namespace ins;

static void separator(const char *title) {
  printf("\n========================================\n");
  printf("  %s\n", title);
  printf("========================================\n");
}

int main() {
  ins::init({"cpu", "cuda"});
  set_device(CPUPlace());

  printf("Insight7 Basic Operations Demo\n");
  printf("OpenBLAS: %s\n", is_compiled_with_openblas() ? "yes" : "no");
  printf("FFTW3:    %s\n", is_compiled_with_fftw3() ? "yes" : "no");

  // --- Array creation ---
  separator("Array Creation");

  Array a = ones({3, 4}, DType::F32, CPUPlace());
  printf("ones(3,4) shape=[%ld,%ld] dtype=%s\n", a.shape().dim(0),
         a.shape().dim(1), dtype_name(a.dtype()));

  Array b = arange(0.0, 12.0, 1.0).to(DType::F32).reshape({3, 4});
  printf("arange(0,12).reshape(3,4):\n%s\n", to_string(b).c_str());

  Array c = eye(4, -1, 0, DType::F64, CPUPlace());
  printf("eye(4) F64:\n%s\n", to_string(c).c_str());

  // --- Multi-dtype arithmetic ---
  separator("Multi-Dtype Arithmetic");

  // F32
  Array f32_a =
      to_array({1.0f, 2.0f, 3.0f}, Shape({3}), DType::F32, CPUPlace());
  Array f32_b =
      to_array({4.0f, 5.0f, 6.0f}, Shape({3}), DType::F32, CPUPlace());
  Array f32_c = f32_a + f32_b;
  printf("F32: [1,2,3] + [4,5,6] = [%g,%g,%g]\n", f32_c.at(0).item<float>(),
         f32_c.at(1).item<float>(), f32_c.at(2).item<float>());

  // F64
  Array f64_a = to_array({1.0, 2.0, 3.0}, Shape({3}), DType::F64, CPUPlace());
  Array f64_b = to_array({0.5, 0.5, 0.5}, Shape({3}), DType::F64, CPUPlace());
  Array f64_c = f64_a * f64_b;
  printf("F64: [1,2,3] * [0.5,0.5,0.5] = [%g,%g,%g]\n",
         f64_c.at(0).item<double>(), f64_c.at(1).item<double>(),
         f64_c.at(2).item<double>());

  // I32
  Array i32_a = to_array({10, 20, 30}, Shape({3}), DType::I32, CPUPlace());
  Array i32_b = to_array({3, 3, 3}, Shape({3}), DType::I32, CPUPlace());
  Array i32_c = i32_a / i32_b;
  printf("I32: [10,20,30] / [3,3,3] = [%d,%d,%d]\n",
         i32_c.at(0).item<int32_t>(), i32_c.at(1).item<int32_t>(),
         i32_c.at(2).item<int32_t>());

  // --- Broadcasting ---
  separator("Broadcasting");

  Array m = to_array({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape({2, 3}), DType::F64,
                     CPUPlace());
  Array row =
      to_array({10.0, 20.0, 30.0}, Shape({1, 3}), DType::F64, CPUPlace());
  Array result = m + row;
  printf("Matrix + Row broadcast:\n%s\n", to_string(result).c_str());

  // --- Unary operations ---
  separator("Unary Operations");

  Array x =
      to_array({-2.0, -1.0, 0.0, 1.0, 2.0}, Shape({5}), DType::F64, CPUPlace());
  printf("x:      %s\n", to_string(x).c_str());
  printf("abs(x): %s\n", to_string(abs(x)).c_str());
  printf("sin(x): %s\n", to_string(sin(x)).c_str());
  printf("exp(x): %s\n", to_string(exp(x)).c_str());

  // --- Reductions ---
  separator("Reductions");

  Array data =
      to_array({1.0, 2.0, 3.0, 4.0, 5.0}, Shape({5}), DType::F64, CPUPlace());
  printf("data:  %s\n", to_string(data).c_str());
  printf("sum:   %g\n", sum(data).item<double>());
  printf("mean:  %g\n", mean(data).item<double>());
  printf("max:   %g\n", max(data).item<double>());
  printf("min:   %g\n", min(data).item<double>());

  printf("\nDone!\n");
  return 0;
}
