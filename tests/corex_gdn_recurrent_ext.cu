#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {

constexpr int kHeadDim = 128;

__global__ void gdn_recurrent_kernel(float* state, const float* query,
                                     const float* key, const float* value,
                                     const float* decay, const float* beta,
                                     float* output, int heads) {
  const int bh = blockIdx.x;
  const int column = threadIdx.x;
  if (column >= kHeadDim) {
    return;
  }

  const int vector_offset = bh * kHeadDim;
  const int state_offset = bh * kHeadDim * kHeadDim;
  const float head_decay = decay[bh];
  float memory = 0.0f;

#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float decayed = state[index] * head_decay;
    state[index] = decayed;
    memory += key[vector_offset + row] * decayed;
  }

  const float delta =
      (value[vector_offset + column] - memory) * beta[bh];
  float result = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float updated =
        state[index] + key[vector_offset + row] * delta;
    state[index] = updated;
    result += query[vector_offset + row] * updated;
  }
  output[vector_offset + column] = result;
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype float32");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor gdn_recurrent_update(torch::Tensor state,
                                   const torch::Tensor& query,
                                   const torch::Tensor& key,
                                   const torch::Tensor& value,
                                   const torch::Tensor& decay,
                                   const torch::Tensor& beta) {
  check_tensor(state, "state");
  check_tensor(query, "query");
  check_tensor(key, "key");
  check_tensor(value, "value");
  check_tensor(decay, "decay");
  check_tensor(beta, "beta");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == kHeadDim &&
                  state.size(3) == kHeadDim,
              "state must have shape (batch, heads, 128, 128)");
  TORCH_CHECK(query.sizes() == value.sizes() && key.sizes() == value.sizes(),
              "query, key, and value shapes must match");
  TORCH_CHECK(query.dim() == 3 && query.size(0) == state.size(0) &&
                  query.size(1) == state.size(1) &&
                  query.size(2) == kHeadDim,
              "query/key/value must have shape (batch, heads, 128)");
  TORCH_CHECK(decay.dim() == 2 && decay.size(0) == state.size(0) &&
                  decay.size(1) == state.size(1) &&
                  beta.sizes() == decay.sizes(),
              "decay/beta must have shape (batch, heads)");

  torch::Tensor output = torch::empty_like(value);
  const int heads = static_cast<int>(state.size(1));
  const int blocks = static_cast<int>(state.size(0) * state.size(1));
  gdn_recurrent_kernel<<<blocks, kHeadDim, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(),
      value.data_ptr<float>(), decay.data_ptr<float>(), beta.data_ptr<float>(),
      output.data_ptr<float>(), heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void gdn_recurrent_update_out(torch::Tensor state,
                              const torch::Tensor& query,
                              const torch::Tensor& key,
                              const torch::Tensor& value,
                              const torch::Tensor& decay,
                              const torch::Tensor& beta,
                              torch::Tensor output) {
  const int blocks = static_cast<int>(state.size(0) * state.size(1));
  gdn_recurrent_kernel<<<blocks, kHeadDim, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(), query.data_ptr<float>(), key.data_ptr<float>(),
      value.data_ptr<float>(), decay.data_ptr<float>(), beta.data_ptr<float>(),
      output.data_ptr<float>(), static_cast<int>(state.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("recurrent_update", &gdn_recurrent_update,
             "Fused CoreX Gated DeltaNet recurrent update");
  module.def("recurrent_update_out", &gdn_recurrent_update_out,
             "Unchecked fused recurrent update with preallocated output");
}
