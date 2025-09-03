#include <iostream>
#include <curand_kernel.h>
#include <format>

// CUDA kernel to count points inside the unit circle
__global__ void monteCarloPi(unsigned int n, unsigned int *count, unsigned int seed) {
  extern __shared__ unsigned int sharedCount[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Initialize shared memory
  sharedCount[tid] = 0;

  // Setup random number generator
  curandState state;
  curand_init(seed, idx, 0, &state);

  // Each thread performs its portion of the simulation
  for (unsigned int i = idx; i < n; i += stride) {
    float x = curand_uniform(&state);
    float y = curand_uniform(&state);
    if (x * x + y * y <= 1.0f)
      sharedCount[tid]++;
  }

  __syncthreads();

  // Reduce shared memory to a single value per block
  if (tid == 0) {
    unsigned int blockSum = 0;
    for (int i = 0; i < blockDim.x; ++i)
      blockSum += sharedCount[i];
    atomicAdd(count, (unsigned int)blockSum);
  }
}

int main() {
  unsigned int N = 1u << 30;  // >1 billion samples
  unsigned int *count;
  cudaMallocManaged(&count, sizeof(unsigned int));
  *count = 0;

  unsigned int blockSize = 1u << 10;  // multiple of 32 (size of a warp)
  unsigned int numBlocks = 1u << 11;

  // Prefetch count to GPU to avoid costly page faults
  cudaMemPrefetchAsync(count, sizeof(unsigned int), 0);

  // Launch kernel with shared memory size = blockSize * sizeof(unsigned int)
  monteCarloPi<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(N, count, time(NULL));
  cudaDeviceSynchronize();

  double pi = 4.0 * (*count) / N;
  std::cout << std::format("Estimated Pi ≈ {}\n", pi);

  cudaFree(count);
  return 0;
}
