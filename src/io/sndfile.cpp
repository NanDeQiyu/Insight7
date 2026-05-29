// src/io/sndfile.cpp
#ifdef INSIGHT_USE_SNDFILE

#include "insight/io/sndfile.h"
#include "insight/core/exception.h"
#include "insight/ops/creation.h"
#include <algorithm>
#include <sndfile.h>
#include <vector>

namespace ins {
namespace audio {

// Detect output format from file extension
static int detect_write_format(const std::string &path) {
  // Find the last dot
  auto dot = path.rfind('.');
  if (dot == std::string::npos) {
    INS_THROW("audio::write: cannot detect format from path '", path,
              "' (no extension)");
  }

  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

  if (ext == "wav")
    return SF_FORMAT_WAV | SF_FORMAT_FLOAT;
  if (ext == "flac")
    return SF_FORMAT_FLAC | SF_FORMAT_PCM_24;
  if (ext == "ogg")
    return SF_FORMAT_OGG | SF_FORMAT_VORBIS;

  INS_THROW("audio::write: unsupported extension '.", ext,
            "'. Supported: .wav, .flac, .ogg");
  return 0; // unreachable
}

std::pair<Array, AudioInfo> read(const std::string &path, const Place &place) {
  SF_INFO sf_info{};
  sf_info.format = 0;

  SNDFILE *file = sf_open(path.c_str(), SFM_READ, &sf_info);
  INS_CHECK(file != nullptr, "audio::read: failed to open '", path,
            "': ", sf_strerror(nullptr));

  AudioInfo info;
  info.frames = sf_info.frames;
  info.samplerate = sf_info.samplerate;
  info.channels = sf_info.channels;
  info.format = sf_info.format;

  int64_t total_samples = sf_info.frames * sf_info.channels;
  std::vector<float> buffer(total_samples);

  sf_count_t read_count = sf_readf_float(file, buffer.data(), sf_info.frames);
  sf_close(file);

  INS_CHECK(read_count == sf_info.frames, "audio::read: expected to read ",
            sf_info.frames, " frames, got ", read_count);

  Shape shape({sf_info.frames, sf_info.channels});
  Array data = to_array<float>(buffer, shape, ins::CPUPlace());

  return {data.to(place), info};
}

void write(const std::string &path, const Array &data, int samplerate) {
  INS_CHECK(data.defined(), "audio::write: input array is undefined");

  int ndim = data.shape().ndim();
  INS_CHECK(ndim == 1 || ndim == 2,
            "audio::write: expected 1D or 2D array, got ", ndim, "D");

  int64_t frames, channels;
  if (ndim == 1) {
    frames = data.numel();
    channels = 1;
  } else {
    frames = data.shape().dim(0);
    channels = data.shape().dim(1);
  }

  INS_CHECK(samplerate > 0, "audio::write: samplerate must be positive, got ",
            samplerate);

  // Convert to contiguous float32 on CPU
  Array float_data = data.to(DType::F32).to(ins::CPUPlace()).contiguous();

  SF_INFO sf_info{};
  sf_info.samplerate = samplerate;
  sf_info.channels = static_cast<int>(channels);
  sf_info.format = detect_write_format(path);

  SNDFILE *file = sf_open(path.c_str(), SFM_WRITE, &sf_info);
  INS_CHECK(file != nullptr, "audio::write: failed to open '", path,
            "': ", sf_strerror(nullptr));

  const float *buf = float_data.data<float>();
  sf_count_t written = sf_writef_float(file, buf, frames);
  sf_close(file);

  INS_CHECK(written == frames, "audio::write: expected to write ", frames,
            " frames, got ", written);
}

} // namespace audio
} // namespace ins

#endif // INSIGHT_USE_SNDFILE
