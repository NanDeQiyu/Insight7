---
name: fix-ci-test-tempdir-race
description: Fix CI test failures from parallel test suites competing for shared temp directories
source: auto-skill
extracted_at: '2026-06-02T00:00:00.000Z'
---

# Fix CI Temp Directory Race Condition

## Symptom

Tests pass locally but fail in CI with:
```
filesystem error: cannot remove: Directory not empty [/tmp/xxx]
```
or:
```
write_bin: cannot open file: /tmp/xxx/test.bin
```

The test itself passes, but `TearDownTestSuite` fails, or a test that depends on `SetUpTestSuite` creating a directory fails because the directory doesn't exist.

## Root Cause

CI runs test suites in parallel. When multiple test suites use the same temp directory path (e.g., `/tmp/insight_io_test`), one suite's `TearDownTestSuite` may remove the directory while another suite's tests are still running.

Even with a single suite, `SetUpTestSuite` runs once per process. If CTest launches the same test binary multiple times (e.g., for different test filters), the second launch may find the directory already removed by the first launch's teardown.

## Fix Pattern

### 1. Create directory in `SetUp()` (per-test), not just `SetUpTestSuite()` (per-suite)

```cpp
// BEFORE (fragile):
class MyTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    std::filesystem::create_directories("/tmp/my_test");
  }
  static void TearDownTestSuite() {
    std::filesystem::remove_all("/tmp/my_test");  // May race!
  }
};

// AFTER (robust):
class MyTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    ins::init({"cpu"});
  }
  static void TearDownTestSuite() {
    std::error_code ec;  // Don't throw if directory already gone
    std::filesystem::remove_all("/tmp/my_test", ec);
  }
  void SetUp() override {
    tmp_dir = "/tmp/my_test";
    std::filesystem::create_directories(tmp_dir);  // Ensure exists before each test
  }
  std::string tmp_dir;
};
```

### 2. Use `error_code` overload for cleanup

```cpp
// BEFORE (throws on missing directory):
std::filesystem::remove_all("/tmp/my_test");

// AFTER (silently handles missing directory):
std::error_code ec;
std::filesystem::remove_all("/tmp/my_test", ec);
```

### 3. Use unique temp paths per test (best)

```cpp
// ✅ Cross-platform: no <unistd.h>, uses std::chrono
#include <chrono>
#include <filesystem>

class MyTest : public ::testing::Test {
protected:
  static std::string tmp_base;
  std::string tmp_dir;

  static void SetUpTestSuite() {
    auto ts = std::chrono::steady_clock::now().time_since_epoch().count();
    tmp_base = (std::filesystem::temp_directory_path() /
                ("insight_test_" + std::to_string(ts)))
                   .string();
    std::filesystem::create_directories(tmp_base);
  }
  static void TearDownTestSuite() {
    std::error_code ec;
    std::filesystem::remove_all(tmp_base, ec);
  }
  void SetUp() override {
    tmp_dir = tmp_base;
    std::filesystem::create_directories(tmp_dir);
  }
};
std::string MyTest::tmp_base;
```

**Why `std::chrono` instead of `getpid()`**: `<unistd.h>` is Linux-only.
`std::filesystem::temp_directory_path()` + `std::chrono::steady_clock` is
cross-platform (C++17). Avoids Windows portability issues.

## Verification

Run the same test binary twice in parallel:
```bash
./tests/insight_tests_cpu --gtest_filter="MyTest.*" &
./tests/insight_tests_cpu --gtest_filter="MyTest.*" &
wait
```

Both should pass without filesystem errors.
