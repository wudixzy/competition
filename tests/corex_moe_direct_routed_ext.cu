#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kTopK = 8;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void direct_w13_kernel(
    const __half* input, const __half* w13, const int64_t* expert_ids,
    __half* gate_up, int rows_per_expert, int hidden) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int total_rows = kTopK * rows_per_expert;
  if (warp >= total_rows) {
    return;
  }

  const int slot = warp / rows_per_expert;
  const int local_row = warp - slot * rows_per_expert;
  const int64_t expert = expert_ids[slot];
  const int64_t weight_row =
      (expert * rows_per_expert + local_row) * static_cast<int64_t>(hidden);
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* weight2 =
      reinterpret_cast<const __half2*>(w13 + weight_row);
  float sum = 0.0f;
  for (int index = lane; index < hidden / 2; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 weight = weight2[index];
    sum = fmaf(__half2float(weight.x), __half2float(x.x), sum);
    sum = fmaf(__half2float(weight.y), __half2float(x.y), sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    gate_up[warp] = __float2half_rn(sum);
  }
}

__global__ void direct_w13_silu_kernel(
    const __half* input, const __half* w13, const int64_t* expert_ids,
    __half* activated, int intermediate, int hidden) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int total_rows = kTopK * intermediate;
  if (warp >= total_rows) {
    return;
  }

  const int slot = warp / intermediate;
  const int local_row = warp - slot * intermediate;
  const int64_t expert = expert_ids[slot];
  const int rows_per_expert = 2 * intermediate;
  const int64_t expert_base =
      expert * rows_per_expert * static_cast<int64_t>(hidden);
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* gate2 = reinterpret_cast<const __half2*>(
      w13 + expert_base + local_row * static_cast<int64_t>(hidden));
  const __half2* up2 = reinterpret_cast<const __half2*>(
      w13 + expert_base
      + (local_row + intermediate) * static_cast<int64_t>(hidden));
  float gate_sum = 0.0f;
  float up_sum = 0.0f;
  for (int index = lane; index < hidden / 2; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 gate = gate2[index];
    const __half2 up = up2[index];
    gate_sum = fmaf(__half2float(gate.x), __half2float(x.x), gate_sum);
    gate_sum = fmaf(__half2float(gate.y), __half2float(x.y), gate_sum);
    up_sum = fmaf(__half2float(up.x), __half2float(x.x), up_sum);
    up_sum = fmaf(__half2float(up.y), __half2float(x.y), up_sum);
  }
  gate_sum = warp_sum(gate_sum);
  up_sum = warp_sum(up_sum);
  if (lane == 0) {
    // Preserve the FP16 W13 boundary before applying the activation.
    const float gate = __half2float(__float2half_rn(gate_sum));
    const float up = __half2float(__float2half_rn(up_sum));
    const float silu = gate / (1.0f + expf(-gate));
    activated[warp] = __float2half_rn(silu * up);
  }
}

__global__ void direct_w2_reduce_kernel(
    const __half* activated, const __half* w2, const int64_t* expert_ids,
    const __half* weights, __half* output, int hidden, int intermediate) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= hidden) {
    return;
  }

  float weighted_sum = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    const int64_t expert = expert_ids[slot];
    const int64_t weight_row =
        (expert * hidden + warp) * static_cast<int64_t>(intermediate);
    const __half2* activation2 = reinterpret_cast<const __half2*>(
        activated + slot * intermediate);
    const __half2* weight2 =
        reinterpret_cast<const __half2*>(w2 + weight_row);
    float expert_sum = 0.0f;
    for (int index = lane; index < intermediate / 2;
         index += kWarpSize) {
      const __half2 x = activation2[index];
      const __half2 weight = weight2[index];
      expert_sum = fmaf(
          __half2float(weight.x), __half2float(x.x), expert_sum);
      expert_sum = fmaf(
          __half2float(weight.y), __half2float(x.y), expert_sum);
    }
    expert_sum = warp_sum(expert_sum);
    if (lane == 0) {
      // Match the existing BMM -> FP16 -> weighted FP16 product boundary.
      const __half expert_half = __float2half_rn(expert_sum);
      const __half product = __hmul(expert_half, weights[slot]);
      weighted_sum += __half2float(product);
    }
  }
  if (lane == 0) {
    output[warp] = __float2half_rn(weighted_sum);
  }
}

__global__ void direct_w2_reduce_serial_half_kernel(
    const __half* activated, const __half* w2, const int64_t* expert_ids,
    const __half* weights, __half* output, int hidden, int intermediate) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= hidden) {
    return;
  }

  __half weighted_sum = __float2half_rn(0.0f);
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    const int64_t expert = expert_ids[slot];
    const int64_t weight_row =
        (expert * hidden + warp) * static_cast<int64_t>(intermediate);
    const __half2* activation2 = reinterpret_cast<const __half2*>(
        activated + slot * intermediate);
    const __half2* weight2 =
        reinterpret_cast<const __half2*>(w2 + weight_row);
    float expert_sum = 0.0f;
    for (int index = lane; index < intermediate / 2;
         index += kWarpSize) {
      const __half2 x = activation2[index];
      const __half2 weight = weight2[index];
      expert_sum = fmaf(
          __half2float(weight.x), __half2float(x.x), expert_sum);
      expert_sum = fmaf(
          __half2float(weight.y), __half2float(x.y), expert_sum);
    }
    expert_sum = warp_sum(expert_sum);
    if (lane == 0) {
      const __half expert_half = __float2half_rn(expert_sum);
      const __half product = __hmul(expert_half, weights[slot]);
      weighted_sum = __hadd(weighted_sum, product);
    }
  }
  if (lane == 0) {
    output[warp] = weighted_sum;
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ids(const torch::Tensor& expert_ids, int64_t experts) {
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");
  TORCH_CHECK(experts > 0, "expert count must be positive");
}

void check_w13_inputs(const torch::Tensor& input,
                      const torch::Tensor& w13,
                      const torch::Tensor& expert_ids) {
  check_half_cuda(input, "input");
  check_half_cuda(w13, "w13");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1,
              "input must have shape (1, hidden)");
  TORCH_CHECK(w13.dim() == 3 && w13.size(2) == input.size(1),
              "w13 must have shape (experts, 2*intermediate, hidden)");
  TORCH_CHECK(w13.size(1) % 2 == 0 && input.size(1) % 2 == 0,
              "W13 dimensions must be even");
  check_ids(expert_ids, w13.size(0));
}

}  // namespace

torch::Tensor direct_w13(const torch::Tensor& input,
                         const torch::Tensor& w13,
                         const torch::Tensor& expert_ids) {
  check_w13_inputs(input, w13, expert_ids);
  const int rows = static_cast<int>(w13.size(1));
  const int hidden = static_cast<int>(w13.size(2));
  auto output = torch::empty({kTopK, rows}, input.options());
  const int warps_per_block = kThreads / kWarpSize;
  const int blocks = (kTopK * rows + warps_per_block - 1) / warps_per_block;
  direct_w13_kernel<<<blocks, kThreads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()), rows, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor direct_w13_silu(const torch::Tensor& input,
                              const torch::Tensor& w13,
                              const torch::Tensor& expert_ids) {
  check_w13_inputs(input, w13, expert_ids);
  const int intermediate = static_cast<int>(w13.size(1) / 2);
  const int hidden = static_cast<int>(w13.size(2));
  auto output = torch::empty({kTopK, intermediate}, input.options());
  const int warps_per_block = kThreads / kWarpSize;
  const int blocks =
      (kTopK * intermediate + warps_per_block - 1) / warps_per_block;
  direct_w13_silu_kernel<<<blocks, kThreads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      intermediate, hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor direct_w2_reduce(const torch::Tensor& activated,
                               const torch::Tensor& w2,
                               const torch::Tensor& expert_ids,
                               const torch::Tensor& weights) {
  check_half_cuda(activated, "activated");
  check_half_cuda(w2, "w2");
  check_half_cuda(weights, "weights");
  TORCH_CHECK(activated.dim() == 2 && activated.size(0) == kTopK,
              "activated must have shape (8, intermediate)");
  TORCH_CHECK(w2.dim() == 3 && w2.size(2) == activated.size(1),
              "w2 must have shape (experts, hidden, intermediate)");
  TORCH_CHECK(weights.dim() == 1 && weights.numel() == kTopK,
              "weights must have shape (8,)");
  TORCH_CHECK(activated.size(1) % 2 == 0,
              "intermediate must be even");
  check_ids(expert_ids, w2.size(0));
  const int hidden = static_cast<int>(w2.size(1));
  const int intermediate = static_cast<int>(w2.size(2));
  auto output = torch::empty({1, hidden}, activated.options());
  const int warps_per_block = kThreads / kWarpSize;
  const int blocks = (hidden + warps_per_block - 1) / warps_per_block;
  direct_w2_reduce_kernel<<<blocks, kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      hidden, intermediate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor direct_w2_reduce_serial_half(
    const torch::Tensor& activated,
    const torch::Tensor& w2,
    const torch::Tensor& expert_ids,
    const torch::Tensor& weights) {
  check_half_cuda(activated, "activated");
  check_half_cuda(w2, "w2");
  check_half_cuda(weights, "weights");
  TORCH_CHECK(activated.dim() == 2 && activated.size(0) == kTopK,
              "activated must have shape (8, intermediate)");
  TORCH_CHECK(w2.dim() == 3 && w2.size(2) == activated.size(1),
              "w2 must have shape (experts, hidden, intermediate)");
  TORCH_CHECK(weights.dim() == 1 && weights.numel() == kTopK,
              "weights must have shape (8,)");
  TORCH_CHECK(activated.size(1) % 2 == 0,
              "intermediate must be even");
  check_ids(expert_ids, w2.size(0));
  const int hidden = static_cast<int>(w2.size(1));
  const int intermediate = static_cast<int>(w2.size(2));
  auto output = torch::empty({1, hidden}, activated.options());
  const int warps_per_block = kThreads / kWarpSize;
  const int blocks = (hidden + warps_per_block - 1) / warps_per_block;
  direct_w2_reduce_serial_half_kernel<<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      hidden, intermediate);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w13", &direct_w13,
             "Direct selected-expert FP16 W13 matvec");
  module.def("w13_silu", &direct_w13_silu,
             "Direct selected-expert W13 matvec with SiLU-and-multiply");
  module.def("w2_reduce", &direct_w2_reduce,
             "Direct selected-expert W2 matvec with routed-weight reduction");
  module.def("w2_reduce_serial_half", &direct_w2_reduce_serial_half,
             "Direct selected-expert W2 with serial FP16 reduction");
}
