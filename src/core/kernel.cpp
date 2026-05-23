// src/core/kernel.cpp
#include "insight/c_api/place.h"
#include "insight/c_api/kernel.h"
#include "insight/c_api/array.h"
#include "insight/c_api/exception.h"
#include "insight/core/place.h"
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <string>
#include <algorithm>

static std::unordered_map<std::string, InsightKernel> g_kernel_registry;
static thread_local std::string kernel_error_message = "";

static std::vector<std::string>& registered_op_names() {
    static std::vector<std::string> names;
    return names;
}

extern "C" {

    C_Status insight_register_kernel(const char* op_name,
        int32_t device_type,
        int32_t dtype,
        InsightKernel kernel) {

        if (!op_name) {
            kernel_error_message = "insight_register_kernel: op_name is null";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        if (!kernel) {
            kernel_error_message = std::string("insight_register_kernel: kernel is null for operator '") + op_name + "'";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        if (device_type != INSIGHT_DEVICE_CPU && device_type != INSIGHT_DEVICE_GPU) {
            kernel_error_message = std::string("insight_register_kernel: invalid device_type ") + std::to_string(device_type)
                + " for operator '" + op_name + "'";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        if (dtype <= INSIGHT_DTYPE_UNKNOWN || dtype >= INSIGHT_DTYPE_COUNT) {
            kernel_error_message = std::string("insight_register_kernel: invalid dtype ") + std::to_string(dtype)
                + " for operator '" + op_name + "'";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        char key[256];
        snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
        g_kernel_registry[key] = kernel;

        auto& names = registered_op_names();
        if (std::find(names.begin(), names.end(), op_name) == names.end()) {
            names.push_back(op_name);
        }

        return C_SUCCESS;
    }

    InsightKernel insight_find_kernel(const char* op_name,
        int32_t device_type,
        int32_t dtype) {
        if (!op_name) return nullptr;

        char key[256];
        snprintf(key, sizeof(key), "%s|%d|%d", op_name, device_type, dtype);
        auto it = g_kernel_registry.find(key);
        return (it != g_kernel_registry.end()) ? it->second : nullptr;
    }

    C_Status insight_kernel_launch(const char* op_name,
        int32_t device_type,
        int32_t dtype,
        void** inputs,
        void** outputs) {
        if (!op_name) {
            kernel_error_message = "insight_kernel_launch: op_name is null";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        if (!inputs) {
            kernel_error_message = std::string("insight_kernel_launch: inputs array is null for operator '") + op_name + "'";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        if (!outputs) {
            kernel_error_message = std::string("insight_kernel_launch: outputs array is null for operator '") + op_name + "'";
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        // Try the requested device first
        InsightKernel kernel = insight_find_kernel(op_name, device_type, dtype);
        if (!kernel) {
            kernel_error_message = std::string("insight_kernel_launch: kernel not found for operator '")
                + op_name + "', device_type=" + std::to_string(device_type)
                + ", dtype=" + std::to_string(dtype);
            insight_set_last_error(kernel_error_message.c_str());
            return C_FAILED;
        }

        C_Status status = kernel(inputs, outputs);

        // Auto fallback to CPU if GPU kernel requests it
        if (status == C_FALLBACK && device_type == INSIGHT_DEVICE_GPU) {
            return insight_kernel_launch(op_name, INSIGHT_DEVICE_CPU, dtype, inputs, outputs);
        }

        if (status != C_SUCCESS && status != C_FALLBACK) {
            kernel_error_message = std::string("Kernel '") + op_name
                + "' failed on device_type=" + std::to_string(device_type)
                + ", dtype=" + std::to_string(dtype)
                + " with status " + std::to_string(status) + "\n";
            kernel_error_message += "Device backend error: ";
            kernel_error_message += ins::Place(static_cast<ins::DeviceKind>(device_type), 0).get_last_error();
            insight_set_last_error(kernel_error_message.c_str());
        }

        return status;
    }

    int insight_has_kernel(const char* op_name,
        int32_t device_type,
        int32_t dtype) {
        return insight_find_kernel(op_name, device_type, dtype) != nullptr ? 1 : 0;
    }

    int insight_get_operator_count(void) {
        return static_cast<int>(registered_op_names().size());
    }

    C_Status insight_get_operator_name(int index, char* buffer, int size) {
        auto& names = registered_op_names();
        if (index < 0 || index >= static_cast<int>(names.size())) {
            return C_FAILED;
        }
        strncpy(buffer, names[index].c_str(), size - 1);
        buffer[size - 1] = '\0';
        return C_SUCCESS;
    }

} // extern "C"