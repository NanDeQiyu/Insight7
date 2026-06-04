// demos/gpu_transfer.cpp
// Demonstrates: GPU device management, CPU-GPU data transfer, zero-copy views
#include "insight/insight.h"
#include <cstdio>
#include <cstring>

using namespace ins;

static void separator(const char *title) {
  printf("\n========================================\n");
  printf("  %s\n", title);
  printf("========================================\n");
}

int main() {
  ins::init();

  if (!ins::has_device(DeviceKind::GPU)) {
    return 0;
  }

  printf("Insight7 GPU Transfer Demo\n");

  // --- CPU to GPU transfer ---
  separator("CPU -> GPU Transfer");

  Array cpu_arr = to_array({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape({2, 3}),
                           DType::F64, CPUPlace());
  printf("CPU array:\n%s\n", to_string(cpu_arr).c_str());

  Array gpu_arr = cpu_arr.to(GPUPlace(0));
  printf("GPU array shape=[%ld,%ld] place=%s\n", gpu_arr.shape().dim(0),
         gpu_arr.shape().dim(1),
         gpu_arr.place().kind() == DeviceKind::GPU ? "gpu" : "cpu");

  // --- GPU to CPU transfer ---
  separator("GPU -> CPU Transfer");

  Array back = gpu_arr.to(CPUPlace());
  printf("Round-tripped back to CPU:\n%s\n", to_string(back).c_str());

  // --- GPU arithmetic ---
  separator("GPU Arithmetic (F64)");

  Array a = to_array({1.0, 2.0, 3.0}, Shape({3}), DType::F64, CPUPlace())
                .to(GPUPlace(0));
  Array b = to_array({4.0, 5.0, 6.0}, Shape({3}), DType::F64, CPUPlace())
                .to(GPUPlace(0));

  Array sum_ab = a + b;
  Array mul_ab = a * b;
  printf("GPU [1,2,3] + [4,5,6] = %s\n",
         to_string(sum_ab.to(CPUPlace())).c_str());
  printf("GPU [1,2,3] * [4,5,6] = %s\n",
         to_string(mul_ab.to(CPUPlace())).c_str());

  // --- GPU arithmetic F32 ---
  separator("GPU Arithmetic (F32)");

  Array a32 = to_array({1.0f, 2.0f, 3.0f}, Shape({3}), DType::F32, CPUPlace())
                  .to(GPUPlace(0));
  Array b32 = to_array({4.0f, 5.0f, 6.0f}, Shape({3}), DType::F32, CPUPlace())
                  .to(GPUPlace(0));
  Array c32 = a32 * b32 + a32;
  printf("GPU F32: [1,2,3]*[4,5,6]+[1,2,3] = %s\n",
         to_string(c32.to(CPUPlace())).c_str());

  // --- GPU reductions ---
  separator("GPU Reductions");

  Array data =
      to_array({1.0, 2.0, 3.0, 4.0, 5.0}, Shape({5}), DType::F64, CPUPlace())
          .to(GPUPlace(0));
  printf("GPU sum:  %g\n", sum(data).to(CPUPlace()).item<double>());
  printf("GPU mean: %g\n", mean(data).to(CPUPlace()).item<double>());
  printf("GPU max:  %g\n", max(data).to(CPUPlace()).item<double>());

  // --- Zero-copy views ---
  separator("Zero-Copy Views (GPU)");

  Array base = arange(0.0, 12.0, 1.0).to(DType::F32).reshape({3, 4});
  Array gpu_base = base.to(GPUPlace(0));

  // Reshape (view, no copy)
  Array reshaped = gpu_base.reshape({4, 3});
  printf("Reshape 3x4 -> 4x3: shape=[%ld,%ld]\n", reshaped.shape().dim(0),
         reshaped.shape().dim(1));

  // Transpose (view, no copy)
  Array transposed = transpose(gpu_base);
  printf("Transpose 3x4 -> 4x3: shape=[%ld,%ld]\n", transposed.shape().dim(0),
         transposed.shape().dim(1));

  // Slice (view, no copy)
  Array sliced = gpu_base.slice(0, 0, 2);
  printf("Slice rows 0-1: shape=[%ld,%ld]\n", sliced.shape().dim(0),
         sliced.shape().dim(1));
  printf("Sliced data: %s\n", to_string(sliced.to(CPUPlace())).c_str());

  printf("\nDone!\n");
  return 0;
}
