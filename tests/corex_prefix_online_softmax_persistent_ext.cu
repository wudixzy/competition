#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr int kTileSize = 512;
constexpr int kThreads = 256;
constexpr int kPersistentBlocks = 1024;

__global__ void persistent_online_softmax_update_kernel(
    float* scores, float* running_max, float* running_sum,
    float* correction, int rows) {
  const int tid = threadIdx.x;
  __shared__ float reduction[kThreads];

  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    float* row_scores = scores + static_cast<int64_t>(row) * kTileSize;
    const int column0 = tid;
    const int column1 = tid + kThreads;

    reduction[tid] = fmaxf(row_scores[column0], row_scores[column1]);
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }

    const float old_max = running_max[row];
    const float new_max = fmaxf(old_max, reduction[0]);
    const float exp0 = expf(row_scores[column0] - new_max);
    const float exp1 = expf(row_scores[column1] - new_max);
    row_scores[column0] = exp0;
    row_scores[column1] = exp1;

    reduction[tid] = exp0 + exp1;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float scale = expf(old_max - new_max);
      correction[row] = scale;
      running_max[row] = new_max;
      running_sum[row] = running_sum[row] * scale + reduction[0];
    }
    __syncthreads();
  }
}

void check_float_cuda_contiguous(const torch::Tensor& tensor,
                                 const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor update_online_softmax_persistent(
    torch::Tensor scores, torch::Tensor running_max,
    torch::Tensor running_sum) {
  check_float_cuda_contiguous(scores, "scores");
  check_float_cuda_contiguous(running_max, "running_max");
  check_float_cuda_contiguous(running_sum, "running_sum");
  TORCH_CHECK(scores.dim() >= 2 && scores.size(-1) == kTileSize,
              "scores must have a final dimension of 512");
  TORCH_CHECK(running_max.sizes() == running_sum.sizes(),
              "running_max and running_sum shapes must match");
  TORCH_CHECK(scores.numel() / kTileSize == running_max.numel(),
              "running state must contain one value per score row");
  TORCH_CHECK(scores.device() == running_max.device()
                  && scores.device() == running_sum.device(),
              "all tensors must be on the same device");

  const int64_t rows64 = running_max.numel();
  TORCH_CHECK(rows64 > 0 && rows64 <= std::numeric_limits<int>::max(),
              "unsupported row count");
  const int rows = static_cast<int>(rows64);
  const int blocks = std::min(rows, kPersistentBlocks);
  auto correction = torch::empty_like(running_max);
  persistent_online_softmax_update_kernel<<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      scores.data_ptr<float>(), running_max.data_ptr<float>(),
      running_sum.data_ptr<float>(), correction.data_ptr<float>(), rows);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return correction;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("update", &update_online_softmax_persistent,
             "Fixed-512 persistent-grid FP32 online-softmax update");
}
