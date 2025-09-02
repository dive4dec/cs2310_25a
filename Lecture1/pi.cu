#include <iostream>
#include <curand_kernel.h>
#include <format>

// CUDA kernel to count points inside the unit circle
__global__ void monteCarloPi(unsigned long long n, unsigned long long *count, unsigned int seed) {
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
  for (unsigned long long i = idx; i < n; i += stride) {
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
    atomicAdd(count, (unsigned long long)blockSum);
  }
}

int main() {
  unsigned long long N = 1ULL << 40;  // >1 trillion samples
  unsigned long long *count;
  cudaMallocManaged(&count, sizeof(unsigned long long));
  *count = 0;

  int blockSize = 1 << 10;
  int numBlocks = 1 << 11;

  // Prefetch count to GPU to avoid costly page faults
  cudaMemPrefetchAsync(count, sizeof(unsigned long long), 0);

  // Launch kernel with shared memory size = blockSize * sizeof(unsigned int)
  monteCarloPi<<<numBlocks, blockSize, blockSize * sizeof(unsigned int)>>>(N, count, time(NULL));
  cudaDeviceSynchronize();

  double pi = 4.0 * (*count) / N;
  std::cout << std::format("Estimated Pi = {}\n", pi);

  cudaFree(count);
  return 0;
}