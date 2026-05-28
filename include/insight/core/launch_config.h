// launch_config.h
#ifndef LAUNCH_CONFIG_H
#define LAUNCH_CONFIG_H
#include <cuda_runtime.h>

class LaunchConfig {
public:
  dim3 blocks;
  dim3 threads;
  int shmSize;
  cudaStream_t stream;

  // default constructor
  LaunchConfig()
      : blocks(1, 1, 1), threads(256, 1, 1), shmSize(0), stream(nullptr) {}

  // Automatically calculate blocks (one-dimensional)
  LaunchConfig(int n, cudaStream_t stream = nullptr)
      : threads(256, 1, 1), shmSize(0), stream(stream) {
    if (n <= 0) {
      blocks = dim3(0, 1, 1);
    } else {
      int num_blocks = (n + threads.x - 1) / threads.x;
      blocks = dim3(num_blocks, 1, 1);
    }
  }

  // Specify threads (one dimension)
  LaunchConfig(int n, int threads_per_block, cudaStream_t stream = nullptr)
      : threads(threads_per_block, 1, 1), shmSize(0), stream(stream) {
    if (threads_per_block <= 0) {
      throw std::invalid_argument("threads_per_block must be positive");
    }
    if (n <= 0) {
      blocks = dim3(0, 1, 1);
    } else {
      int num_blocks = (n + threads_per_block - 1) / threads_per_block;
      blocks = dim3(num_blocks, 1, 1);
    }
  }

  // Specify threads and shared memory (one-dimensional)
  LaunchConfig(int n, int threads_per_block, int shm_size,
               cudaStream_t stream = nullptr)
      : threads(threads_per_block, 1, 1), shmSize(shm_size), stream(stream) {
    if (threads_per_block <= 0) {
      throw std::invalid_argument("threads_per_block must be positive");
    }
    if (n <= 0) {
      blocks = dim3(0, 1, 1);
    } else {
      int num_blocks = (n + threads_per_block - 1) / threads_per_block;
      blocks = dim3(num_blocks, 1, 1);
    }
  }

  // New: 2D constructor
  LaunchConfig(dim3 grid, dim3 block, int shm_size = 0,
               cudaStream_t stream = nullptr)
      : blocks(grid), threads(block), shmSize(shm_size), stream(stream) {}

  // Set method compatible with old code (one-dimensional)
  void set(int blocks, int threads, int shmSize, cudaStream_t stream) {
    if (threads <= 0) {
      throw std::invalid_argument("threads must be positive");
    }
    this->blocks = dim3(blocks, 1, 1);
    this->threads = dim3(threads, 1, 1);
    this->shmSize = shmSize;
    this->stream = stream;
  }

  // New: 2D set method
  void set(dim3 blocks, dim3 threads, int shmSize, cudaStream_t stream) {
    this->blocks = blocks;
    this->threads = threads;
    this->shmSize = shmSize;
    this->stream = stream;
  }
};

#endif // LAUNCH_CONFIG_H