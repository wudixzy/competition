#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

__global__ void add_one_kernel(float* values, int64_t count) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] += 1.0f;
  }
}

torch::Tensor add_one(torch::Tensor input) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  torch::Tensor output = input.clone();
  constexpr int threads = 256;
  const int64_t count = output.numel();
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  add_one_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<float>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("add_one", &add_one, "CoreX Torch extension smoke kernel");
}
