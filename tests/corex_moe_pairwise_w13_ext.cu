#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kHidden = 2048;
constexpr int kRowsPerExpert = 256;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline void accumulate_half2(float& sum, const __half2 value,
                                        const __half2 weight) {
  sum = fmaf(__half2float(weight.x), __half2float(value.x), sum);
  sum = fmaf(__half2float(weight.y), __half2float(value.y), sum);
}

__global__ void pairwise_w13_kernel(
    const __half* input, const __half* w13, const int64_t* expert_ids,
    __half* output) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= kTopK * kRowsPerExpert) {
    return;
  }

  const int slot = warp / kRowsPerExpert;
  const int row = warp - slot * kRowsPerExpert;
  const int64_t expert = expert_ids[slot];
  const int64_t weight_row =
      (expert * kRowsPerExpert + row) * static_cast<int64_t>(kHidden);
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* weight2 =
      reinterpret_cast<const __half2*>(w13 + weight_row);

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;
  constexpr int kHalf2Count = kHidden / 2;
  constexpr int kStride = 4 * kWarpSize;
  for (int base = lane; base < kHalf2Count; base += kStride) {
    accumulate_half2(sum0, input2[base], weight2[base]);
    accumulate_half2(
        sum1, input2[base + kWarpSize], weight2[base + kWarpSize]);
    accumulate_half2(
        sum2, input2[base + 2 * kWarpSize],
        weight2[base + 2 * kWarpSize]);
    accumulate_half2(
        sum3, input2[base + 3 * kWarpSize],
        weight2[base + 3 * kWarpSize]);
  }
  const float lane_sum = (sum0 + sum1) + (sum2 + sum3);
  const float total = warp_sum(lane_sum);
  if (lane == 0) {
    output[warp] = __float2half_rn(total);
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor pairwise_w13(const torch::Tensor& input,
                           const torch::Tensor& w13,
                           const torch::Tensor& expert_ids) {
  check_half_cuda(input, "input");
  check_half_cuda(w13, "w13");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({1, kHidden}),
              "input must have shape (1, 2048)");
  TORCH_CHECK(
      w13.sizes()
          == torch::IntArrayRef({kExperts, kRowsPerExpert, kHidden}),
      "w13 must have shape (256, 256, 2048)");
  TORCH_CHECK(
      expert_ids.sizes() == torch::IntArrayRef({kTopK}),
      "expert_ids must have shape (8,)");

  auto output = torch::empty(
      {kTopK, kRowsPerExpert}, input.options());
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  constexpr int kBlocks =
      (kTopK * kRowsPerExpert + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pairwise_w13_kernel<<<kBlocks, kThreads, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w13", &pairwise_w13,
             "Pairwise-accumulated direct selected-expert FP16 W13");
}
