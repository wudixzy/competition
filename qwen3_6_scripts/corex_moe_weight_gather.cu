#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kGridCap = 1024;

__global__ void selected_weight_gather_half2_kernel(
    const __half2* w13, const __half2* w2, const int64_t* expert_ids,
    __half2* selected_w13, __half2* selected_w2,
    int64_t w13_pairs_per_expert, int64_t w2_pairs_per_expert) {
  const int64_t w13_total = kTopK * w13_pairs_per_expert;
  const int64_t total = w13_total + kTopK * w2_pairs_per_expert;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    if (index < w13_total) {
      const int slot = index / w13_pairs_per_expert;
      const int64_t local = index - slot * w13_pairs_per_expert;
      selected_w13[index] =
          w13[expert_ids[slot] * w13_pairs_per_expert + local];
    } else {
      const int64_t output_index = index - w13_total;
      const int slot = output_index / w2_pairs_per_expert;
      const int64_t local = output_index - slot * w2_pairs_per_expert;
      selected_w2[output_index] =
          w2[expert_ids[slot] * w2_pairs_per_expert + local];
    }
  }
}

void check_weight(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank three");
  TORCH_CHECK(tensor.size(2) % 2 == 0,
              name, " innermost dimension must be even");
}

}  // namespace

std::vector<torch::Tensor> gather_selected_weights(
    const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& expert_ids) {
  check_weight(w13, "w13");
  check_weight(w2, "w2");
  TORCH_CHECK(w13.device() == w2.device(),
              "W13/W2 must be on the same device");
  TORCH_CHECK(w13.size(0) == w2.size(0),
              "W13/W2 expert counts differ");
  TORCH_CHECK(w13.size(2) == w2.size(1),
              "W13/W2 hidden dimensions differ");
  TORCH_CHECK(w13.size(1) == 2 * w2.size(2),
              "W13/W2 intermediate dimensions differ");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.device() == w13.device(),
              "weights and expert_ids must be on the same device");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");

  auto selected_w13 = torch::empty(
      {kTopK, w13.size(1), w13.size(2)}, w13.options());
  auto selected_w2 = torch::empty(
      {kTopK, w2.size(1), w2.size(2)}, w2.options());
  const int64_t w13_pairs_per_expert = w13.size(1) * w13.size(2) / 2;
  const int64_t w2_pairs_per_expert = w2.size(1) * w2.size(2) / 2;
  const int64_t total =
      kTopK * (w13_pairs_per_expert + w2_pairs_per_expert);
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kGridCap));
  selected_weight_gather_half2_kernel<<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half2*>(w13.data_ptr<at::Half>()),
      reinterpret_cast<const __half2*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half2*>(selected_w13.data_ptr<at::Half>()),
      reinterpret_cast<__half2*>(selected_w2.data_ptr<at::Half>()),
      w13_pairs_per_expert, w2_pairs_per_expert);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {selected_w13, selected_w2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather", &gather_selected_weights,
             "Gather selected FP16 top-8 MoE weights with half2 loads");
}
