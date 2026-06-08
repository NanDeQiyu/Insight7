// tests/cpu/cpp/test_profiler.cpp
#include "insight/c_api/place.h"
#include "insight/c_api/profiler.h"
#include "insight/core/profiler.h"
#include "insight/insight.h"
#include "gtest/gtest.h"
#include <thread>

class ProfilerTestCPU : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// ========== C API Tests ==========

TEST_F(ProfilerTestCPU, TimerCreateDestroy) {
  InsightPlace cpu_place = {INSIGHT_DEVICE_CPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&cpu_place, &timer), C_SUCCESS);
  ASSERT_NE(timer, nullptr);
  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestCPU, TimerStartStopElapsed) {
  InsightPlace cpu_place = {INSIGHT_DEVICE_CPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&cpu_place, &timer), C_SUCCESS);

  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);

  float ms = 0.0f;
  ASSERT_EQ(insight_timer_elapsed_ms(timer, &ms), C_SUCCESS);
  EXPECT_GE(ms, 5.0f);
  EXPECT_LE(ms, 50.0f);

  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestCPU, TimerReset) {
  InsightPlace cpu_place = {INSIGHT_DEVICE_CPU, 0};
  InsightTimer timer = nullptr;
  ASSERT_EQ(insight_timer_create(&cpu_place, &timer), C_SUCCESS);

  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);
  float ms1 = 0.0f;
  insight_timer_elapsed_ms(timer, &ms1);

  ASSERT_EQ(insight_timer_reset(timer), C_SUCCESS);
  ASSERT_EQ(insight_timer_start(timer), C_SUCCESS);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_EQ(insight_timer_stop(timer), C_SUCCESS);
  float ms2 = 0.0f;
  ASSERT_EQ(insight_timer_elapsed_ms(timer, &ms2), C_SUCCESS);
  EXPECT_GE(ms2, 5.0f);

  ASSERT_EQ(insight_timer_destroy(timer), C_SUCCESS);
}

TEST_F(ProfilerTestCPU, TimerNullInput) {
  EXPECT_NE(insight_timer_create(nullptr, nullptr), C_SUCCESS);
  EXPECT_NE(insight_timer_destroy(nullptr), C_SUCCESS);
  EXPECT_NE(insight_timer_start(nullptr), C_SUCCESS);
  EXPECT_NE(insight_timer_stop(nullptr), C_SUCCESS);
  EXPECT_NE(insight_timer_elapsed_ms(nullptr, nullptr), C_SUCCESS);
}

// ========== C++ Timer Class Tests ==========

TEST_F(ProfilerTestCPU, CppTimerCreate) {
  ins::Timer timer(ins::CPUPlace());
  EXPECT_FALSE(timer.started());
}

TEST_F(ProfilerTestCPU, CppTimerMeasure) {
  ins::Timer timer(ins::CPUPlace());
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  timer.stop();
  float ms = timer.elapsed_ms();
  EXPECT_GE(ms, 0.0f);
  EXPECT_LE(ms, 50.0f);
}

TEST_F(ProfilerTestCPU, CppTimerMove) {
  ins::Timer t1(ins::CPUPlace());
  ins::Timer t2(std::move(t1));
  EXPECT_FALSE(t2.started());
}

TEST_F(ProfilerTestCPU, CppTimerReset) {
  ins::Timer timer(ins::CPUPlace());
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  timer.stop();
  float ms1 = timer.elapsed_ms();

  timer.reset();
  timer.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  timer.stop();
  float ms2 = timer.elapsed_ms();

  EXPECT_GE(ms2, 0.0f);
}
