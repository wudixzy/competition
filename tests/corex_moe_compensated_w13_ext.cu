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

__device__ inline float subtract_rn(float left, float right) {
  return __fadd_rn(left, -right);
}

__device__ inline void compensated_add(float value, float& sum,
                                        float& correction) {
  const float adjusted = subtract_rn(value, correction);
  const float next = __fadd_rn(sum, adjusted);
  correction = subtract_rn(subtract_rn(next, sum), adjusted);
  sum = next;
}

__device__ inline float warp_sum_rn(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    const float peer = __shfl_down_sync(0xffffffff, value, offset);
    value = __fadd_rn(value, peer);
  }
  return value;
}

__global__ void compensated_w13_kernel(
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

  float sum = 0.0f;
  float correction = 0.0f;
  for (int index = lane; index < kHidden / 2; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 weight = weight2[index];
    compensated_add(
        __fmul_rn(__half2float(weight.x), __half2float(x.x)),
        sum, correction);
    compensated_add(
        __fmul_rn(__half2float(weight.y), __half2float(x.y)),
        sum, correction);
  }

  const float total = warp_sum_rn(sum);
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

torch::Tensor compensated_w13(const torch::Tensor& input,
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

  auto output = torch::empty({kTopK, kRowsPerExpert}, input.options());
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  constexpr int kBlocks =
      (kTopK * kRowsPerExpert + kWarpsPerBlock - 1) / kWarpsPerBlock;
  compensated_w13_kernel<<<kBlocks, kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w13", &compensated_w13,
             "Kahan-compensated selected-expert FP16 W13 matvec");
}
