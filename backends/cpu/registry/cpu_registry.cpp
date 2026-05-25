// backends/cpu/registry/cpu_registry.cpp
#include "cpu_registry.h"
#include "insight/c_api/place.h"
#include <cstdio>
#include <string>
#include <unordered_map>

// Meyer's Singleton - guaranteed to be initialized on first use
static std::unordered_map<std::string, InsightKernel> &get_cpu_kernel_map() {
  static std::unordered_map<std::string, InsightKernel> map;
  return map;
}

void cpu_register_kernel(const char *op_name, int32_t dtype,
                         InsightKernel func) {
  char key[256];
  snprintf(key, sizeof(key), "%s|%d", op_name, dtype);
  get_cpu_kernel_map()[key] = func;
}

void cpu_sync_kernels(C_Status (*register_fn)(const char *, int32_t, int32_t,
                                              InsightKernel)) {
  for (const auto &[key, kernel] : get_cpu_kernel_map()) {
    char op_name[128];
    int32_t dtype;
    sscanf(key.c_str(), "%[^|]|%d", op_name, &dtype);
    register_fn(op_name, INSIGHT_DEVICE_CPU, dtype, kernel);
  }
}