// src/core/kernel.cpp
#include "insight/c_api/kernel.h"
#include "insight/c_api/array.h"
#include "insight/c_api/exception.h"
#include "insight/c_api/place.h"
#include "insight/core/place.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static std::unordered_map<std::string, InsightKernel> g_kernel_registry;
static thread_local std::string kernel_error_message = "";

static std::vector<std::string> &registered_op_names() {
  static std::vector<std::string> names;
  return names;
}

// Count non-NULL entries in a NULL-terminated void** array.
static int count_ptrs(void **ptrs) {
  if (!ptrs)
    return 0;
  int n = 0;
  while (ptrs[n])
    ++n;
  return n;
}

// ============================================================================
// CPU fallback with GPU→CPU data transfer
// ============================================================================

static C_Status do_cpu_fallback(const char *op_name, int32_t dtype,
                                void **inputs, void **outputs) {
  InsightKernel cpu_kernel =
      insight_find_kernel(op_name, INSIGHT_DEVICE_CPU, dtype);
  if (!cpu_kernel) {
    kernel_error_message =
        std::string("insight_kernel_launch: CPU fallback kernel not found "
                    "for operator '") +
        op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  int num_inputs = count_ptrs(inputs);
  int num_outputs = count_ptrs(outputs);

  ins::Place cpu_place = ins::CPUPlace(0);
  ins::Place gpu_place = ins::GPUPlace(0);

  struct Transfer {
    InsightArray *arr;
    bool is_output;
  };
  std::vector<Transfer> transfers;
  std::vector<InsightArray *> seen;

  auto consider = [&](void *ptr, bool is_output) {
    if (!ptr)
      return;
    InsightArray *arr = (InsightArray *)ptr;
    // Must be a valid InsightArray on GPU, not a scalar pointer.
    if (arr->dtype <= INSIGHT_DTYPE_UNKNOWN ||
        arr->dtype >= INSIGHT_DTYPE_COUNT)
      return;
    if (arr->device_type != INSIGHT_DEVICE_GPU)
      return;
    if (arr->ndim < 0 || arr->ndim > INSIGHT_MAX_NDIM)
      return;
    if (arr->numel < 0)
      return;
    // Deduplicate: same InsightArray may appear in both inputs and outputs.
    for (size_t i = 0; i < seen.size(); ++i) {
      if (seen[i] == arr) {
        if (is_output)
          transfers[i].is_output = true;
        return;
      }
    }
    seen.push_back(arr);
    transfers.push_back({arr, is_output});
  };

  for (int i = 0; i < num_inputs; ++i)
    consider(inputs[i], false);
  for (int i = 0; i < num_outputs; ++i)
    consider(outputs[i], true);

  // GPU → CPU
  std::vector<void *> orig_ptrs(transfers.size());
  for (size_t t = 0; t < transfers.size(); ++t) {
    InsightArray *a = transfers[t].arr;
    size_t bytes = (size_t)a->numel * insight_dtype_size(a->dtype);
    void *cpu = malloc(bytes);
    if (!cpu) {
      // Rollback
      for (size_t j = 0; j < t; ++j) {
        free(transfers[j].arr->data);
        transfers[j].arr->data = orig_ptrs[j];
        transfers[j].arr->device_type = INSIGHT_DEVICE_GPU;
        transfers[j].arr->device_id = gpu_place.device_id();
      }
      kernel_error_message = "insight_kernel_launch: CPU malloc failed";
      insight_set_last_error(kernel_error_message.c_str());
      return C_FAILED;
    }
    gpu_place.copy_to_host(cpu, a->data, bytes);
    orig_ptrs[t] = a->data;
    a->data = cpu;
    a->device_type = INSIGHT_DEVICE_CPU;
    a->device_id = 0;
  }

  // Run CPU kernel
  C_Status status = cpu_kernel(inputs, outputs);

  if (status != C_SUCCESS) {
    const char *err = cpu_place.get_last_error();
    if (err && err[0] != '\0')
      insight_set_last_error(err);
  }

  // CPU → GPU (outputs only) and restore
  for (size_t t = 0; t < transfers.size(); ++t) {
    InsightArray *a = transfers[t].arr;
    size_t bytes = (size_t)a->numel * insight_dtype_size(a->dtype);
    if (transfers[t].is_output && status == C_SUCCESS)
      gpu_place.copy_from_host(orig_ptrs[t], a->data, bytes);
    free(a->data);
    a->data = orig_ptrs[t];
    a->device_type = INSIGHT_DEVICE_GPU;
    a->device_id = gpu_place.device_id();
  }

  gpu_place.synchronize();
  return status;
}

// ============================================================================
// insight_kernel_launch — unified entry point
// ============================================================================

extern "C" {

C_Status insight_register_kernel(const char *op_name, int32_t device_type,
                                 int32_t dtype, InsightKernel kernel) {
  if (!op_name) {
    kernel_error_message = "insight_register_kernel: op_name is null";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (!kernel) {
    kernel_error_message =
        std::string("insight_register_kernel: kernel is null for '") + op_name +
        "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (device_type != INSIGHT_DEVICE_CPU && device_type != INSIGHT_DEVICE_GPU) {
    kernel_error_message =
        std::string("insight_register_kernel: invalid device_type ") +
        std::to_string(device_type) + " for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (dtype <= INSIGHT_DTYPE_UNKNOWN || dtype >= INSIGHT_DTYPE_COUNT) {
    kernel_error_message =
        std::string("insight_register_kernel: invalid dtype ") +
        std::to_string(dtype) + " for '" + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  char key[256];
  snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
  g_kernel_registry[key] = kernel;

  auto &names = registered_op_names();
  if (std::find(names.begin(), names.end(), op_name) == names.end())
    names.push_back(op_name);
  return C_SUCCESS;
}

InsightKernel insight_find_kernel(const char *op_name, int32_t device_type,
                                  int32_t dtype) {
  if (!op_name)
    return nullptr;
  char key[256];
  snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
  auto it = g_kernel_registry.find(key);
  return (it != g_kernel_registry.end()) ? it->second : nullptr;
}

C_Status insight_kernel_launch(const char *op_name, int32_t device_type,
                               int32_t dtype, void **inputs, void **outputs) {
  if (!op_name) {
    kernel_error_message = "insight_kernel_launch: op_name is null";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }
  if (!inputs || !outputs) {
    kernel_error_message =
        std::string("insight_kernel_launch: null array for '") + op_name + "'";
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  InsightKernel kernel = insight_find_kernel(op_name, device_type, dtype);
  if (!kernel) {
    kernel_error_message =
        std::string("insight_kernel_launch: kernel not found for '") + op_name +
        "', device=" + std::to_string(device_type) +
        ", dtype=" + std::to_string(dtype);
    insight_set_last_error(kernel_error_message.c_str());
    return C_FAILED;
  }

  C_Status status = kernel(inputs, outputs);

  if (status == C_FALLBACK && device_type == INSIGHT_DEVICE_GPU)
    return do_cpu_fallback(op_name, dtype, inputs, outputs);

  if (status != C_SUCCESS && status != C_FALLBACK) {
    kernel_error_message =
        std::string("Kernel '") + op_name +
        "' failed (device=" + std::to_string(device_type) +
        ", dtype=" + std::to_string(dtype) +
        ", status=" + std::to_string(status) + ")\nBackend: " +
        ins::Place(static_cast<ins::DeviceKind>(device_type), 0)
            .get_last_error();
    insight_set_last_error(kernel_error_message.c_str());
  }
  return status;
}

int insight_has_kernel(const char *op_name, int32_t device_type,
                       int32_t dtype) {
  return insight_find_kernel(op_name, device_type, dtype) != nullptr ? 1 : 0;
}

int insight_get_operator_count(void) {
  return static_cast<int>(registered_op_names().size());
}

C_Status insight_get_operator_name(int index, char *buffer, int size) {
  auto &names = registered_op_names();
  if (index < 0 || index >= static_cast<int>(names.size()))
    return C_FAILED;
  strncpy(buffer, names[index].c_str(), size - 1);
  buffer[size - 1] = '\0';
  return C_SUCCESS;
}

} // extern "C"
