#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 128;
constexpr int kW13Rows = 2 * kIntermediate;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void dual_w13_sum_kernel(
    const __half* input, const __half* w13, const int64_t* expert_ids,
    float* forward_sums, float* reverse_sums) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= kTopK * kW13Rows) {
    return;
  }

  const int slot = warp / kW13Rows;
  const int local_row = warp - slot * kW13Rows;
  const int64_t expert = expert_ids[slot];
  const int64_t weight_row =
      (expert * kW13Rows + local_row) * static_cast<int64_t>(kHidden);
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* weight2 =
      reinterpret_cast<const __half2*>(w13 + weight_row);

  float forward = 0.0f;
  float reverse = 0.0f;
  constexpr int kPairs = kHidden / 2;
  for (int index = lane; index < kPairs; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 weight = weight2[index];
    forward = fmaf(__half2float(weight.x), __half2float(x.x), forward);
    forward = fmaf(__half2float(weight.y), __half2float(x.y), forward);

    const int reverse_index = kPairs - 1 - index;
    const __half2 reverse_x = input2[reverse_index];
    const __half2 reverse_weight = weight2[reverse_index];
    reverse = fmaf(__half2float(reverse_weight.y),
                   __half2float(reverse_x.y), reverse);
    reverse = fmaf(__half2float(reverse_weight.x),
                   __half2float(reverse_x.x), reverse);
  }

  forward = warp_sum(forward);
  reverse = warp_sum(reverse);
  if (lane == 0) {
    forward_sums[warp] = forward;
    reverse_sums[warp] = reverse;
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ids(const torch::Tensor& expert_ids) {
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");
}

}  // namespace

std::vector<torch::Tensor> dual_w13_sums(
    const torch::Tensor& input, const torch::Tensor& w13,
    const torch::Tensor& expert_ids) {
  check_half_cuda(input, "input");
  check_half_cuda(w13, "w13");
  check_ids(expert_ids);
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1
                  && input.size(1) == kHidden,
              "input must have shape (1, 2048)");
  TORCH_CHECK(w13.dim() == 3 && w13.size(0) == kExperts
                  && w13.size(1) == kW13Rows
                  && w13.size(2) == kHidden,
              "w13 must have shape (256, 256, 2048)");

  auto options = input.options().dtype(torch::kFloat32);
  auto forward = torch::empty({kTopK, kW13Rows}, options);
  auto reverse = torch::empty({kTopK, kW13Rows}, options);
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  constexpr int kBlocks =
      (kTopK * kW13Rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  dual_w13_sum_kernel<<<kBlocks, kThreads, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(), forward.data_ptr<float>(),
      reverse.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {forward, reverse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("dual_w13_sums", &dual_w13_sums,
             "Forward and reverse FP32 W13 reductions");
}
