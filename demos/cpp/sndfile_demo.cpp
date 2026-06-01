// demos/sndfile_demo.cpp
//
// Audio processing demo: read a real song, analyze its spectrum,
// apply a low-pass filter, and write the result.
//
// Usage: ./demo_sndfile
// Output: /tmp/insight_demo_*.wav

#ifdef INSIGHT_USE_SNDFILE

#include "insight/core/array.h"
#include "insight/init.h"
#include "insight/io/sndfile.h"
#include "insight/ops/creation.h"
#include "insight/ops/elementwise.h"
#include "insight/ops/fft.h"
#include "insight/ops/operator.h"
#include "insight/ops/reduction.h"
#include "insight/ops/signal.h"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace ins;

static const std::string TMP = "/tmp/insight_demo_";

// Generate a multi-frequency test signal
static Array generate_test_signal(int64_t frames, float sample_rate) {
  std::vector<float> data(frames);
  for (int64_t i = 0; i < frames; ++i) {
    float t = static_cast<float>(i) / sample_rate;
    data[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t) +
              0.3f * std::sin(2.0f * M_PI * 880.0f * t) +
              0.1f * std::sin(2.0f * M_PI * 3141.0f * t);
  }
  return to_array<float>(data);
}

int main() {
  ins::init({"cpu"});

  std::cout << "=== Insight Audio Processing Demo ===" << std::endl;
  std::cout << std::endl;

  // Try to read the real audio file first
  const std::string audio_path =
      AUDIO_DIR "/大梦何时了-叶炫清-210356449-2000.flac";

  Array signal;
  int sample_rate;
  int64_t frames;

  try {
    std::cout << "[0] Reading: " << audio_path << std::endl;
    auto [data, info] = audio::read(audio_path);
    sample_rate = info.samplerate;
    frames = info.frames;

    std::cout << "    Format: FLAC, " << info.channels << "ch, "
              << info.samplerate << "Hz, " << info.frames << " frames ("
              << static_cast<float>(info.frames) / info.samplerate << "s)"
              << std::endl;

    // Mix to mono by averaging channels
    if (info.channels == 2) {
      // data shape: [frames, 2]
      Array left = data.slice(1, 0, 1).squeeze(1);  // [frames]
      Array right = data.slice(1, 1, 2).squeeze(1); // [frames]
      signal = (left + right) * 0.5f;
    } else {
      signal = data.squeeze(1); // [frames]
    }
  } catch (const std::exception &e) {
    std::cout << "    Could not read FLAC file: " << e.what() << std::endl;
    std::cout << "    Generating synthetic signal instead..." << std::endl;
    sample_rate = 44100;
    frames = static_cast<int64_t>(sample_rate * 2);
    signal = generate_test_signal(frames, static_cast<float>(sample_rate));
  }

  // Only process first 5 seconds to keep it fast
  int64_t max_frames = static_cast<int64_t>(sample_rate) * 5;
  if (frames > max_frames) {
    std::cout << "    Truncating to first 5 seconds (" << max_frames
              << " frames)" << std::endl;
    signal = signal.slice(0, 0, max_frames);
    frames = max_frames;
  }

  std::cout << std::endl;

  // Step 1: Write mono signal
  std::cout << "[1] Writing mono signal..." << std::endl;
  Array signal_2d = signal.reshape({frames, 1});
  audio::write(TMP + "original.wav", signal_2d, sample_rate);
  std::cout << "    Written: " << TMP << "original.wav" << std::endl;

  // Step 2: Low-pass filter via FFT
  std::cout << "[2] Applying low-pass filter (cutoff 1000Hz)..." << std::endl;
  Array spectrum = fft::rfft(signal);

  int64_t freq_bins = spectrum.numel();
  int64_t cutoff_bin =
      static_cast<int64_t>(1000.0 / (sample_rate / 2.0) * freq_bins);
  int64_t taper_width = freq_bins / 100;

  // Build a complex mask: pass low frequencies, taper at cutoff
  std::vector<std::complex<float>> mask_data(freq_bins, {0.0f, 0.0f});
  for (int64_t i = 0; i < cutoff_bin - taper_width / 2; ++i) {
    mask_data[i] = {1.0f, 1.0f};
  }
  for (int64_t i = std::max(0L, cutoff_bin - taper_width / 2);
       i < cutoff_bin + taper_width / 2 && i < freq_bins; ++i) {
    float t = 1.0f - static_cast<float>(i - cutoff_bin + taper_width / 2) /
                         taper_width;
    float v = std::max(0.0f, std::min(1.0f, t));
    mask_data[i] = {v, v};
  }

  Array mask = to_array<std::complex<float>>(mask_data, spectrum.shape(),
                                             ins::CPUPlace());
  Array filtered_spectrum = mul(spectrum, mask);
  Array filtered = fft::irfft(filtered_spectrum, frames);

  // Reshape to [frames, 1] for write
  Array filtered_2d = filtered.reshape({frames, 1});
  audio::write(TMP + "filtered.wav", filtered_2d, sample_rate);
  std::cout << "    Written: " << TMP << "filtered.wav" << std::endl;

  // Step 3: Energy comparison
  std::cout << std::endl;
  std::cout << "[3] Signal statistics:" << std::endl;
  Array energy_before = sum(mul(signal, signal));
  Array energy_after = sum(mul(filtered, filtered));
  float eb = energy_before.item<float>();
  float ea = energy_after.item<float>();
  std::cout << "    Energy (original):  " << eb << std::endl;
  std::cout << "    Energy (filtered):  " << ea << std::endl;
  if (eb > 0) {
    std::cout << "    Retained: " << (ea / eb) * 100.0f << "%" << std::endl;
  }

  std::cout << std::endl;
  std::cout << "Output files:" << std::endl;
  std::cout << "  " << TMP << "original.wav  - Mono mixdown" << std::endl;
  std::cout << "  " << TMP << "filtered.wav  - Low-pass < 1000Hz" << std::endl;

  return 0;
}

#else

#include <iostream>

int main() {
  std::cerr << "This demo requires libsndfile (INSIGHT_USE_SNDFILE)."
            << std::endl;
  return 1;
}

#endif
