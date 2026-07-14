#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int kHeadDim = 128;

__global__ void gdn_recurrent_update_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ beta,
    const float* __restrict__ decay,
    float* __restrict__ state,
    float* __restrict__ output) {
  const int head = blockIdx.x;
  const int column = threadIdx.x;

  __shared__ float shared_query[kHeadDim];
  __shared__ float shared_key[kHeadDim];
  shared_query[column] = query[head * kHeadDim + column];
  shared_key[column] = key[head * kHeadDim + column];
  __syncthreads();

  const int64_t state_base =
      static_cast<int64_t>(head) * kHeadDim * kHeadDim;
  const float head_decay = decay[head];

  float key_state = 0.0F;
#pragma unroll 4
  for (int row = 0; row < kHeadDim; ++row) {
    const float decayed =
        state[state_base + row * kHeadDim + column] * head_decay;
    key_state = fmaf(shared_key[row], decayed, key_state);
  }

  const float delta =
      (value[head * kHeadDim + column] - key_state) * beta[head];
  float head_output = 0.0F;
#pragma unroll 4
  for (int row = 0; row < kHeadDim; ++row) {
    const int64_t offset = state_base + row * kHeadDim + column;
    const float decayed = state[offset] * head_decay;
    const float updated = fmaf(shared_key[row], delta, decayed);
    state[offset] = updated;
    head_output = fmaf(shared_query[row], updated, head_output);
  }
  output[head * kHeadDim + column] = head_output;
}

void check_float_cuda_contiguous(
    const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on a CUDA device");
  TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor gdn_recurrent_update_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor beta,
    torch::Tensor decay,
    torch::Tensor state) {
  check_float_cuda_contiguous(query, "query");
  check_float_cuda_contiguous(key, "key");
  check_float_cuda_contiguous(value, "value");
  check_float_cuda_contiguous(beta, "beta");
  check_float_cuda_contiguous(decay, "decay");
  check_float_cuda_contiguous(state, "state");

  TORCH_CHECK(query.dim() == 3, "query must have shape (B, H, 128)");
  TORCH_CHECK(query.sizes() == key.sizes(), "query/key shapes must match");
  TORCH_CHECK(query.sizes() == value.sizes(), "query/value shapes must match");
  TORCH_CHECK(query.size(2) == kHeadDim, "head dimension must be 128");
  TORCH_CHECK(state.dim() == 4, "state must have shape (B, H, 128, 128)");
  TORCH_CHECK(state.size(0) == query.size(0), "state batch mismatch");
  TORCH_CHECK(state.size(1) == query.size(1), "state head mismatch");
  TORCH_CHECK(
      state.size(2) == kHeadDim && state.size(3) == kHeadDim,
      "state dimensions must both be 128");
  TORCH_CHECK(
      beta.numel() == query.size(0) * query.size(1),
      "beta must have one value per head");
  TORCH_CHECK(decay.numel() == beta.numel(), "decay shape must match beta");

  auto output = torch::empty_like(value);
  const int heads = query.size(0) * query.size(1);
  const auto stream = at::cuda::getCurrentCUDAStream(query.get_device());
  gdn_recurrent_update_kernel<<<heads, kHeadDim, 0, stream>>>(
      query.data_ptr<float>(),
      key.data_ptr<float>(),
      value.data_ptr<float>(),
      beta.data_ptr<float>(),
      decay.data_ptr<float>(),
      state.data_ptr<float>(),
      output.data_ptr<float>());
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
  return output;
}
