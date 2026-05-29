// tests/cpu/test_audio.cpp
#ifdef INSIGHT_USE_SNDFILE

#include "insight/core/array.h"
#include "insight/init.h"
#include "insight/io/sndfile.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/reduction.h"
#include "insight/utils/features.h"
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ins;

class AudioTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() { ins::init(); }
};

// Generate a sine wave as a float32 array on CPU
static Array make_sine(int64_t frames, int channels, float freq,
                       float sample_rate) {
  std::vector<float> data(frames * channels);
  for (int64_t i = 0; i < frames; ++i) {
    float val = std::sin(2.0f * M_PI * freq * i / sample_rate);
    for (int ch = 0; ch < channels; ++ch) {
      data[i * channels + ch] = val;
    }
  }
  Shape shape({frames, channels});
  return to_array<float>(data, shape, ins::CPUPlace());
}

static const std::string tmp_path(const std::string &name,
                                  const std::string &ext = ".wav") {
  return "/tmp/insight_test_audio_" + name + ext;
}

// ========== Round-trip tests ==========

TEST_F(AudioTest, ReadWriteMono) {
  std::string path = tmp_path("mono");
  int64_t frames = 1024;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  audio::write(path, original, 44100);

  auto [read_back, info] = audio::read(path);
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);
  EXPECT_EQ(read_back.shape().dim(0), frames);
  EXPECT_EQ(read_back.shape().dim(1), 1);

  Array diff_arr = abs(sub(read_back, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTest, ReadWriteStereo) {
  std::string path = tmp_path("stereo");
  int64_t frames = 2048;
  Array original = make_sine(frames, 2, 880.0f, 44100.0f);

  audio::write(path, original, 48000);

  auto [read_back, info] = audio::read(path);
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 48000);
  EXPECT_EQ(info.channels, 2);

  Array diff_arr = abs(sub(read_back, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTest, WriteMono1D) {
  std::string path = tmp_path("mono1d");
  int64_t frames = 512;

  std::vector<float> data(frames);
  for (int64_t i = 0; i < frames; ++i)
    data[i] = std::sin(2.0f * M_PI * 220.0f * i / 44100.0f);
  Array original = to_array<float>(data, ins::CPUPlace());

  audio::write(path, original, 44100);

  auto [read_back, info] = audio::read(path);
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.channels, 1);

  Array read_flat = read_back.reshape({frames});
  Array diff_arr = abs(sub(read_flat, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-6f);

  std::remove(path.c_str());
}

TEST_F(AudioTest, ReadNonexistentThrows) {
  EXPECT_THROW(audio::read("/tmp/nonexistent_audio_file_xyz.wav"),
               std::exception);
}

TEST_F(AudioTest, WriteInvalidDimsThrows) {
  Array bad = zeros({2, 3, 4}, DType::F32, ins::CPUPlace());
  EXPECT_THROW(audio::write(tmp_path("bad"), bad), std::exception);
}

TEST_F(AudioTest, WriteReadFlac) {
  std::string path = tmp_path("flac_roundtrip", ".flac");
  int64_t frames = 2048;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  audio::write(path, original, 44100);

  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);

  // FLAC is lossless but 24-bit quantization adds small error
  Array diff_arr = abs(sub(read_back, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-5f);

  std::remove(path.c_str());
}

TEST_F(AudioTest, WriteReadOgg) {
  std::string path = tmp_path("ogg_roundtrip", ".ogg");
  int64_t frames = 4096;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);

  audio::write(path, original, 44100);

  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);
  EXPECT_EQ(info.samplerate, 44100);
  EXPECT_EQ(info.channels, 1);

  // OGG Vorbis is lossy, allow larger tolerance
  Array diff_arr = abs(sub(read_back, original));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 0.05f);

  std::remove(path.c_str());
}

TEST_F(AudioTest, ReadWithExplicitCPUPlace) {
  std::string path = tmp_path("place_cpu");
  int64_t frames = 512;
  Array original = make_sine(frames, 1, 440.0f, 44100.0f);
  audio::write(path, original, 44100);

  auto [read_back, info] = audio::read(path, ins::CPUPlace());
  EXPECT_EQ(read_back.place(), ins::CPUPlace());
  EXPECT_EQ(info.frames, frames);

  std::remove(path.c_str());
}

TEST_F(AudioTest, FFTRoundTrip) {
  if (!ins::is_compiled_with_fftw3())
    GTEST_SKIP() << "FFTW3 not available";
  int64_t frames = 4096;
  Array signal = make_sine(frames, 1, 440.0f, 44100.0f);
  Array signal_1d = signal.reshape({frames});

  Array spectrum = fft::rfft(signal_1d);
  Array reconstructed = fft::irfft(spectrum, frames);

  Array diff_arr = abs(sub(reconstructed, signal_1d));
  Array max_diff = max(diff_arr);
  EXPECT_LT(max_diff.item<float>(), 1e-3f);
}

#endif // INSIGHT_USE_SNDFILE
