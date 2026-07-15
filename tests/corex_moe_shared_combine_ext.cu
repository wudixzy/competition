#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__global__ void shared_combine_kernel(const half* routed,
                                      const half* shared,
                                      const half* gate,
                                      half* output, int elements,
                                      int hidden) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int batch = index / hidden;
  const float gate_value = __half2float(gate[batch]);
  const half sigmoid_gate = __float2half(
      1.0f / (1.0f + expf(-gate_value)));
  const volatile half gated_shared = __float2half_rn(
      __half2float(shared[index]) * __half2float(sigmoid_gate));
  output[index] = __float2half_rn(
      __half2float(routed[index]) + __half2float(gated_shared));
}

__global__ void shared_combine_gate_kernel(const half* routed,
                                           const half* shared,
                                           const half* sigmoid_gate,
                                           half* output, int elements,
                                           int hidden) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const int batch = index / hidden;
  const volatile half gated_shared = __float2half_rn(
      __half2float(shared[index]) * __half2float(sigmoid_gate[batch]));
  output[index] = __float2half_rn(
      __half2float(routed[index]) + __half2float(gated_shared));
}

void check_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must have shape (batch, hidden)");
}

}  // namespace

torch::Tensor shared_combine(const torch::Tensor& routed,
                             const torch::Tensor& shared,
                             const torch::Tensor& gate) {
  check_matrix(routed, "routed");
  check_matrix(shared, "shared");
  check_matrix(gate, "gate");
  TORCH_CHECK(routed.sizes() == shared.sizes(),
              "routed and shared shapes must match");
  TORCH_CHECK(gate.size(0) == routed.size(0) && gate.size(1) == 1,
              "gate must have shape (batch, 1)");

  torch::Tensor output = torch::empty_like(routed);
  const int hidden = static_cast<int>(routed.size(1));
  const int elements = static_cast<int>(routed.numel());
  constexpr int threads = 256;
  const int blocks = (elements + threads - 1) / threads;
  shared_combine_kernel<<<blocks, threads, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(routed.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(shared.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), elements, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor shared_combine_gate(const torch::Tensor& routed,
                                  const torch::Tensor& shared,
                                  const torch::Tensor& sigmoid_gate) {
  check_matrix(routed, "routed");
  check_matrix(shared, "shared");
  check_matrix(sigmoid_gate, "sigmoid_gate");
  TORCH_CHECK(routed.sizes() == shared.sizes(),
              "routed and shared shapes must match");
  TORCH_CHECK(sigmoid_gate.size(0) == routed.size(0)
                  && sigmoid_gate.size(1) == 1,
              "sigmoid_gate must have shape (batch, 1)");

  torch::Tensor output = torch::empty_like(routed);
  const int hidden = static_cast<int>(routed.size(1));
  const int elements = static_cast<int>(routed.numel());
  constexpr int threads = 256;
  const int blocks = (elements + threads - 1) / threads;
  shared_combine_gate_kernel<<<blocks, threads, 0,
                               at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(routed.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(shared.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(sigmoid_gate.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), elements, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("shared_combine", &shared_combine,
             "Exact fused MoE shared gate and routed add");
  module.def("shared_combine_gate", &shared_combine_gate,
             "Fuse MoE shared multiply and routed add after sigmoid");
}
