// src/init.cpp
#include "insight/init.h"
#include "insight/c_api/device_ext.h"
#include "insight/c_api/kernel.h"
#include "insight/core/exception.h"
#include "insight/core/place.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
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
static std::vector<std::string> g_extra_search_paths;

void add_backend_search_path(const std::string &path) {
  g_extra_search_paths.push_back(path);
}

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

static std::vector<std::string> scan_dir_for_backends(const char *dir_path) {
  std::vector<std::string> found;
  const char *prefix = "libinsight_";
  const char *suffix = "_backend";
  const char *ext = backend_extension();
  size_t prefix_len = strlen(prefix);
  size_t suffix_len = strlen(suffix);
  size_t ext_len = strlen(ext);

#ifdef _WIN32
  std::string pattern = std::string(dir_path) + "\\insight_*_backend.dll";
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
  if (h != INVALID_HANDLE_VALUE) {
    do {
      std::string name(fd.cFileName);
      size_t p = name.find("insight_");
      size_t e = name.rfind("_backend.dll");
      if (p != std::string::npos && e != std::string::npos && e > p + 8) {
        found.push_back(name.substr(p + 8, e - p - 8));
      }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
  }
#else
  DIR *dir = opendir(dir_path);
  if (!dir)
    return found;
  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
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

  return found;
}

static std::vector<std::string> discover_backends() {
  std::vector<std::string> found;

  // 1. Current directory
  auto cwd = scan_dir_for_backends(".");
  found.insert(found.end(), cwd.begin(), cwd.end());

  // 2. Extra search paths (registered by language bindings)
  for (const auto &path : g_extra_search_paths) {
    auto extra_found = scan_dir_for_backends(path.c_str());
    found.insert(found.end(), extra_found.begin(), extra_found.end());
  }

  // 3. LD_LIBRARY_PATH (Linux/macOS) or PATH (Windows)
#ifdef _WIN32
  const char *env_path = getenv("PATH");
  char path_sep = ';';
#else
  const char *env_path = getenv("LD_LIBRARY_PATH");
  char path_sep = ':';
#endif
  if (env_path) {
    std::string ld(env_path);
    std::istringstream ss(ld);
    std::string dir;
    while (std::getline(ss, dir, path_sep)) {
      if (!dir.empty()) {
        auto ld_found = scan_dir_for_backends(dir.c_str());
        found.insert(found.end(), ld_found.begin(), ld_found.end());
      }
    }
  }

  std::sort(found.begin(), found.end());
  found.erase(std::unique(found.begin(), found.end()), found.end());
  return found;
}

// ========================================================================
// Backend loading
// ========================================================================

static bool try_load_backend(DeviceKind kind, const char *lib_name) {
  std::string lib_filename =
      std::string(backend_prefix()) + lib_name + backend_extension();

  // Try default search (LD_LIBRARY_PATH, system paths)
  LibHandle lib = load_library(lib_filename.c_str());

  // Try extra search paths (e.g. Python package directory, build dir)
  if (!lib) {
    for (const auto &path : g_extra_search_paths) {
#ifdef _WIN32
      std::string full = path + "\\" + lib_filename;
#else
      std::string full = path + "/" + lib_filename;
#endif
      lib = load_library(full.c_str());
      if (lib)
        break;
    }
  }

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

  // Initialize the default device (e.g. create CUDA context).
  // Without this, the first GPU operation fails because no context exists.
  if (params.interface->init_device) {
    C_Device_st dev;
    std::memset(&dev, 0, sizeof(dev));
    params.interface->init_device(&dev);
  }

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
    // 3. If no GPU backend found via scan, scan environment path directories
    //    for any insight_*_backend.dll/.so (cuda, npu, rocm, etc.)
    if (!get_device_interface(DeviceKind::GPU)) {
#ifdef _WIN32
      const char *env_path = getenv("PATH");
      char path_sep = ';';
#else
      const char *env_path = getenv("LD_LIBRARY_PATH");
      char path_sep = ':';
#endif
      if (env_path) {
        std::string ld(env_path);
        std::istringstream ss(ld);
        std::string dir;
        while (std::getline(ss, dir, path_sep)) {
          if (dir.empty())
            continue;
#ifdef _WIN32
          // Windows: use scan_dir_for_backends which handles FindFirstFileA
          auto gpu_found = scan_dir_for_backends(dir.c_str());
          for (const auto &name : gpu_found) {
            if (name == "cpu")
              continue;
            if (try_load_backend(DeviceKind::GPU,
                                 ("insight_" + name + "_backend").c_str())) {
              goto gpu_found;
            }
          }
#else
          auto *dp = opendir(dir.c_str());
          if (!dp)
            continue;
          struct dirent *entry;
          while ((entry = readdir(dp)) != nullptr) {
            std::string name(entry->d_name);
            // Match libinsight_*_backend.so (any GPU backend)
            if (name.find("libinsight_") == 0 &&
                name.find("_backend.so") != std::string::npos &&
                name.find("_cpu_") == std::string::npos) {
              std::string backend_name =
                  name.substr(11, name.find("_backend") - 11);
              if (try_load_backend(
                      DeviceKind::GPU,
                      ("insight_" + backend_name + "_backend").c_str())) {
                closedir(dp);
                goto gpu_found;
              }
            }
          }
          closedir(dp);
#endif
        }
      gpu_found:;
      }
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
