// src/init.cpp
#include "insight/init.h"
#include "insight/c_api/device_ext.h"
#include "insight/c_api/kernel.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <dlfcn.h>
#endif

#ifdef interface
#undef interface
#endif

namespace ins {

static bool g_initialized = false;

// ========================================================================
// Dynamic library loading helpers
// ========================================================================

#ifdef _WIN32
using LibHandle = HMODULE;

static LibHandle load_library(const char *path) {
  LibHandle handle = LoadLibraryA(path);
  if (!handle) {
    DWORD error = GetLastError();
    char *error_msg = nullptr;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   (LPSTR)&error_msg, 0, NULL);
    if (error_msg)
      LocalFree(error_msg);
  }
  return handle;
}

static void *get_symbol(LibHandle lib, const char *name) {
  return reinterpret_cast<void *>(GetProcAddress(lib, name));
}

static void unload_library(LibHandle lib) { FreeLibrary(lib); }
static const char *backend_prefix() { return ""; }
static const char *backend_extension() { return ".dll"; }
#else
using LibHandle = void *;

static LibHandle load_library(const char *path) {
  LibHandle handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
  return handle;
}

static void *get_symbol(LibHandle lib, const char *name) {
  return dlsym(lib, name);
}

static void unload_library(LibHandle lib) { dlclose(lib); }
static const char *backend_prefix() { return "lib"; }
static const char *backend_extension() { return ".so"; }
#endif

// ========================================================================
// Backend discovery — scan current directory for libinsight_*_backend.so
// ========================================================================

static std::vector<std::string> discover_backends() {
  std::vector<std::string> found;
  const char *prefix = "libinsight_";
  const char *suffix = "_backend";
  const char *ext = backend_extension();
  size_t prefix_len = strlen(prefix);
  size_t suffix_len = strlen(suffix);
  size_t ext_len = strlen(ext);

#ifdef _WIN32
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA("insight_*_backend.dll", &fd);
  if (h != INVALID_HANDLE_VALUE) {
    do {
      std::string name(fd.cFileName);
      // Extract backend name: "insight_<name>_backend.dll"
      size_t p = name.find("insight_");
      size_t e = name.rfind("_backend.dll");
      if (p != std::string::npos && e != std::string::npos && e > p + 8) {
        found.push_back(name.substr(p + 8, e - p - 8));
      }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
  }
#else
  DIR *dir = opendir(".");
  if (!dir)
    return found;
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    // Match libinsight_*_backend.so
    if (name.size() > prefix_len + suffix_len + ext_len &&
        name.compare(0, prefix_len, prefix) == 0 &&
        name.compare(name.size() - ext_len, ext_len, ext) == 0) {
      size_t start = prefix_len;
      size_t end = name.size() - ext_len - suffix_len;
      if (end > start) {
        found.push_back(name.substr(start, end - start));
      }
    }
  }
  closedir(dir);
#endif

  std::sort(found.begin(), found.end());
  return found;
}

// ========================================================================
// Backend loading
// ========================================================================

static bool try_load_backend(DeviceKind kind, const char *lib_name) {
  char lib_path[512];
  snprintf(lib_path, sizeof(lib_path), "%s%s%s", backend_prefix(), lib_name,
           backend_extension());

  LibHandle lib = load_library(lib_path);
  if (!lib)
    return false;

  const char *entry_name =
      (kind == DeviceKind::GPU) ? "InitPluginGPU" : "InitPluginCPU";
  typedef C_Status (*InitPluginFunc)(CustomRuntimeParams *);
  InitPluginFunc init_plugin =
      reinterpret_cast<InitPluginFunc>(get_symbol(lib, entry_name));
  if (!init_plugin) {
    unload_library(lib);
    return false;
  }

  CustomRuntimeParams params;
  std::memset(&params, 0, sizeof(params));
  params.size = sizeof(CustomRuntimeParams);
  params.interface = new C_DeviceInterface();
  params.interface->size = sizeof(C_DeviceInterface);
  params.register_kernel = insight_register_kernel;

  C_Status status = init_plugin(&params);
  if (status != C_SUCCESS) {
    delete params.interface;
    unload_library(lib);
    return false;
  }

  set_device_interface(kind, params.interface, params.device_type);
  return true;
}

// ========================================================================
// Public API
// ========================================================================

void init(std::optional<std::vector<std::string>> backends) {
  if (g_initialized)
    return;

  if (!backends.has_value()) {
    // === Auto-discover mode (default) ===
    // 1. Load CPU backend (required)
    if (!try_load_backend(DeviceKind::CPU, "insight_cpu_backend")) {
      INS_THROW("Failed to load CPU backend");
    }
    // 2. Scan current directory for other backends
    auto available = discover_backends();
    for (const auto &name : available) {
      if (name == "cpu")
        continue;
      if (try_load_backend(DeviceKind::GPU,
                           ("insight_" + name + "_backend").c_str())) {
        break;
      }
    }
    // 3. If no GPU backend found via scan, try common names directly
    //    (dlopen uses LD_LIBRARY_PATH, works when .so is in another directory
    //     e.g. luarocks install, pip install, or system packages)
    if (!get_device_interface(DeviceKind::GPU)) {
      try_load_backend(DeviceKind::GPU, "insight_cuda_backend");
    }
  } else if (backends->empty()) {
    // === Empty vector: load nothing ===
  } else {
    // === Specified backends: load each in order ===
    for (const auto &backend : *backends) {
      if (backend == "cpu") {
        if (!get_device_interface(DeviceKind::CPU)) {
          if (!try_load_backend(DeviceKind::CPU, "insight_cpu_backend")) {
            INS_THROW("Failed to load CPU backend");
          }
        }
      } else {
        if (!get_device_interface(DeviceKind::GPU)) {
          std::string lib_name = "insight_" + backend + "_backend";
          if (!try_load_backend(DeviceKind::GPU, lib_name.c_str())) {
            INS_THROW("Failed to load GPU backend: " + backend);
          }
        }
      }
    }
  }

  g_initialized = true;
}

void init(const std::vector<std::string> &backends) {
  init(std::optional<std::vector<std::string>>(backends));
}

bool is_initialized() { return g_initialized; }

bool has_device(DeviceKind kind) {
  return get_device_interface(kind) != nullptr;
}

void load_backend(const std::string &backend) {
  if (backend == "cpu") {
    if (!get_device_interface(DeviceKind::CPU)) {
      try_load_backend(DeviceKind::CPU, "insight_cpu_backend");
    }
  } else {
    if (!get_device_interface(DeviceKind::GPU)) {
      std::string lib_name = "insight_" + backend + "_backend";
      try_load_backend(DeviceKind::GPU, lib_name.c_str());
    }
  }
}

} // namespace ins
