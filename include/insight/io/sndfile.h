// insight/io/sndfile.h
#pragma once
#include "insight/core/array.h"
#include "insight/core/place.h"
#include <cstdint>
#include <string>

namespace ins {
namespace audio {

/**
 * @brief Audio file metadata.
 */
struct AudioInfo {
  int64_t frames; ///< Number of audio frames (samples per channel)
  int samplerate; ///< Sample rate in Hz (e.g. 44100)
  int channels;   ///< Number of audio channels (1=mono, 2=stereo)
  int format;     ///< libsndfile format bitmask
};

/**
 * @brief Read an audio file into an Array.
 *
 * The returned array has shape [frames, channels] (row-major, channel-last).
 * For mono files the shape is [frames, 1].
 * Data is always converted to float32 internally.
 *
 * @param path   Path to the audio file (WAV, FLAC, OGG, etc.)
 * @param place  Device placement (default: current device)
 * @return A pair of (data, info).
 *
 * @throws std::runtime_error if the file cannot be opened or read.
 */
std::pair<Array, AudioInfo> read(const std::string &path,
                                 const Place &place = get_device());

/**
 * @brief Write an Array to an audio file.
 *
 * The output format is determined by the file extension:
 *   .wav  → WAV (PCM float)
 *   .flac → FLAC (24-bit PCM)
 *   .ogg  → Ogg Vorbis
 *
 * Input array must be 2D [frames, channels] or 1D [frames] (mono).
 * Float data is expected in [-1, 1] range.
 *
 * @param path       Output file path
 * @param data       Audio data array
 * @param samplerate Sample rate in Hz (default: 44100)
 *
 * @throws std::runtime_error if the file cannot be written or format
 *         is unsupported.
 */
void write(const std::string &path, const Array &data, int samplerate = 44100);

} // namespace audio
} // namespace ins
