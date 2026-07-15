#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;

__global__ void exact_reduce_kernel(const __half* expert_output,
                                    const __half* weights,
                                    __half* output, int hidden) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= hidden) {
    return;
  }
  float sum = 0.0f;
#pragma unroll
  for (int expert = 0; expert < kTopK; ++expert) {
    const __half product = __hmul(
        expert_output[expert * hidden + column], weights[expert]);
    sum += __half2float(product);
  }
  output[column] = __float2half_rn(sum);
}

void check_input(const torch::Tensor& expert_output,
                 const torch::Tensor& weights) {
  TORCH_CHECK(expert_output.is_cuda() && weights.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(expert_output.scalar_type() == torch::kFloat16
                  && weights.scalar_type() == torch::kFloat16,
              "inputs must have dtype float16");
  TORCH_CHECK(expert_output.is_contiguous() && weights.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(expert_output.dim() == 2
                  && expert_output.size(0) == kTopK,
              "expert_output must have shape (8, hidden)");
  TORCH_CHECK(weights.dim() == 1 && weights.size(0) == kTopK,
              "weights must have shape (8,)");
}

}  // namespace

torch::Tensor exact_reduce(const torch::Tensor& expert_output,
                           const torch::Tensor& weights) {
  check_input(expert_output, weights);
  auto output = torch::empty(
      {1, expert_output.size(1)}, expert_output.options());
  const int hidden = static_cast<int>(expert_output.size(1));
  const int blocks = (hidden + kThreads - 1) / kThreads;
  exact_reduce_kernel<<<blocks, kThreads, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(expert_output.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("exact_reduce", &exact_reduce,
             "Exact CoreX MoE weighted reduction");
}
