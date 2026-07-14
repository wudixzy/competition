#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void add_one(float* values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] += 1.0f;
  }
}

int main() {
  constexpr int count = 256;
  float host[count];
  for (int index = 0; index < count; ++index) {
    host[index] = static_cast<float>(index);
  }

  float* device = nullptr;
  if (cudaMalloc(&device, sizeof(host)) != cudaSuccess ||
      cudaMemcpy(device, host, sizeof(host), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    std::fprintf(stderr, "allocation or H2D copy failed\n");
    return 1;
  }

  add_one<<<1, count>>>(device, count);
  const cudaError_t launch_status = cudaGetLastError();
  const cudaError_t copy_status =
      cudaMemcpy(host, device, sizeof(host), cudaMemcpyDeviceToHost);
  cudaFree(device);
  if (launch_status != cudaSuccess || copy_status != cudaSuccess) {
    std::fprintf(stderr, "kernel or D2H copy failed: %s / %s\n",
                 cudaGetErrorString(launch_status),
                 cudaGetErrorString(copy_status));
    return 2;
  }

  for (int index = 0; index < count; ++index) {
    if (std::fabs(host[index] - static_cast<float>(index + 1)) > 1e-6f) {
      std::fprintf(stderr, "mismatch at %d: %.6f\n", index, host[index]);
      return 3;
    }
  }
  std::printf("COREX_EXTENSION_SMOKE_OK count=%d last=%.1f\n", count,
              host[count - 1]);
  return 0;
}
