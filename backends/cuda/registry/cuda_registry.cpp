// backends/cuda/registry/cuda_registry.cpp
#include "cuda_registry.h"
#include "insight/c_api/place.h"
#include <string>
#include <unordered_map>
#include <cstdio>

// Meyer's Singleton - guaranteed to be initialized on first use
static std::unordered_map<std::string, InsightKernel>& get_gpu_kernel_map() {
    static std::unordered_map<std::string, InsightKernel> map;
    return map;
}

void gpu_register_kernel(const char* op_name, int32_t dtype, InsightKernel func) {
    char key[256];
    snprintf(key, sizeof(key), "%s|%d", op_name, dtype);
    get_gpu_kernel_map()[key] = func;
}

void gpu_sync_kernels(C_Status(*register_fn)(const char*, int32_t, int32_t, InsightKernel)) {
    for (const auto& [key, kernel] : get_gpu_kernel_map()) {
        char op_name[128];
        int32_t dtype;
        sscanf(key.c_str(), "%[^|]|%d", op_name, &dtype);
        register_fn(op_name, INSIGHT_DEVICE_GPU, dtype, kernel);
    }
}