// demos/cpp/timer_demo.cpp
/// Timer Demo — Verifies the Timer API works correctly.
#include "insight/core/profiler.h"
#include "insight/insight.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include <chrono>
#include <iostream>
#include <thread>

int main() {
  ins::init();

  // CPU Timer
  {
    ins::Timer timer(ins::CPUPlace());
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    timer.stop();
    float ms = timer.elapsed_ms();
    std::cout << "CPU timer: " << ms << " ms" << std::endl;
    if (ms < 0.0f || ms > 100.0f) {
      std::cerr << "ERROR: CPU timer out of range: " << ms << " ms"
                << std::endl;
      return 1;
    }
  }

  // GPU Timer (if available)
  if (ins::is_device_available(ins::DeviceKind::GPU)) {
    ins::Timer timer(ins::GPUPlace(0));
    timer.start();
    ins::Array a = ins::ones({512, 512}, ins::DType::F32, ins::GPUPlace(0));
    ins::Array b =
        ins::full({512, 512}, 2.0f, ins::DType::F32, ins::GPUPlace(0));
    ins::Array c = ins::add(a, b);
    timer.stop();
    float ms = timer.elapsed_ms();
    std::cout << "GPU timer: " << ms << " ms" << std::endl;
    if (ms < 0.0f) {
      std::cerr << "ERROR: GPU timer returned negative: " << ms << " ms"
                << std::endl;
      return 1;
    }
  } else {
    std::cout << "GPU timer: SKIPPED (GPU not available)" << std::endl;
  }

  std::cout << "OK" << std::endl;
  return 0;
}
