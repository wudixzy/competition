#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <vector>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kGridX = 8;

template <int kUnroll>
__global__ void selected_weight_gather_unroll_kernel(
    const uint4* w13, const uint4* w2, const int64_t* expert_ids,
    uint4* selected_w13, uint4* selected_w2,
    int64_t w13_vecs_per_expert, int64_t w2_vecs_per_expert) {
  const int segment = blockIdx.y;
  const int slot = segment & (kTopK - 1);
  const bool copy_w2 = segment >= kTopK;
  const int64_t count =
      copy_w2 ? w2_vecs_per_expert : w13_vecs_per_expert;
  const uint4* source = copy_w2 ? w2 : w13;
  uint4* output = copy_w2 ? selected_w2 : selected_w13;
  const int64_t source_offset = expert_ids[slot] * count;
  const int64_t output_offset = static_cast<int64_t>(slot) * count;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t base =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       base < count; base += stride * kUnroll) {
#pragma unroll
    for (int item = 0; item < kUnroll; ++item) {
      const int64_t index = base + static_cast<int64_t>(item) * stride;
      if (index < count) {
        output[output_offset + index] = source[source_offset + index];
      }
    }
  }
}

void check_weight(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3, name, " must be rank three");
  TORCH_CHECK(tensor.size(1) * tensor.size(2) % 8 == 0,
              name, " expert slices must be divisible by 16 bytes");
}

template <int kUnroll>
void launch(const torch::Tensor& w13, const torch::Tensor& w2,
            const torch::Tensor& expert_ids, torch::Tensor& selected_w13,
            torch::Tensor& selected_w2) {
  const int64_t w13_vecs_per_expert = w13.size(1) * w13.size(2) / 8;
  const int64_t w2_vecs_per_expert = w2.size(1) * w2.size(2) / 8;
  const dim3 grid(kGridX, 2 * kTopK);
  selected_weight_gather_unroll_kernel<kUnroll><<<
      grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint4*>(w13.data_ptr<at::Half>()),
      reinterpret_cast<const uint4*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<uint4*>(selected_w13.data_ptr<at::Half>()),
      reinterpret_cast<uint4*>(selected_w2.data_ptr<at::Half>()),
      w13_vecs_per_expert, w2_vecs_per_expert);
}

}  // namespace

std::vector<torch::Tensor> gather_selected_weights_unroll(
    const torch::Tensor& w13, const torch::Tensor& w2,
    const torch::Tensor& expert_ids, int64_t unroll) {
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
  TORCH_CHECK(unroll == 1 || unroll == 2 || unroll == 4 || unroll == 8,
              "unroll must be one of 1, 2, 4, or 8");

  const int64_t w13_numel = kTopK * w13.size(1) * w13.size(2);
  const int64_t w2_numel = kTopK * w2.size(1) * w2.size(2);
  auto storage = torch::empty({w13_numel + w2_numel}, w13.options());
  auto selected_w13 = storage.narrow(0, 0, w13_numel).view(
      {kTopK, w13.size(1), w13.size(2)});
  auto selected_w2 = storage.narrow(0, w13_numel, w2_numel).view(
      {kTopK, w2.size(1), w2.size(2)});
  switch (unroll) {
    case 1:
      launch<1>(w13, w2, expert_ids, selected_w13, selected_w2);
      break;
    case 2:
      launch<2>(w13, w2, expert_ids, selected_w13, selected_w2);
      break;
    case 4:
      launch<4>(w13, w2, expert_ids, selected_w13, selected_w2);
      break;
    case 8:
      launch<8>(w13, w2, expert_ids, selected_w13, selected_w2);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {selected_w13, selected_w2};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("gather", &gather_selected_weights_unroll,
             "Packed-output selected-weight gather with copy unrolling");
}
