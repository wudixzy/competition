#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
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

__global__ void gdn_prep_recurrent_kernel(
    float* state, const half* raw_query, const half* raw_key,
    const float* value, const float* decay, const float* beta, float* output,
    int value_heads, int key_heads, int expand_ratio) {
  const int bh = blockIdx.x;
  const int column = threadIdx.x;
  const int batch = bh / value_heads;
  const int value_head = bh % value_heads;
  const int key_head = value_head / expand_ratio;
  const int raw_offset = (batch * key_heads + key_head) * kHeadDim;
  const int vector_offset = bh * kHeadDim;
  const int state_offset = bh * kHeadDim * kHeadDim;

  __shared__ float norm_squares[kHeadDim * 2];
  const float raw_q = __half2float(raw_query[raw_offset + column]);
  const float raw_k = __half2float(raw_key[raw_offset + column]);
  norm_squares[column] = raw_q * raw_q;
  norm_squares[kHeadDim + column] = raw_k * raw_k;
  __syncthreads();
  for (int stride = kHeadDim / 2; stride > 0; stride >>= 1) {
    if (column < stride) {
      norm_squares[column] += norm_squares[column + stride];
      norm_squares[kHeadDim + column] +=
          norm_squares[kHeadDim + column + stride];
    }
    __syncthreads();
  }
  const float q_scale = rsqrtf(norm_squares[0] + 1e-6f) * 0.0883883476483f;
  const float k_scale = rsqrtf(norm_squares[kHeadDim] + 1e-6f);
  const float head_decay = decay[bh];
  float memory = 0.0f;

#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float decayed = state[index] * head_decay;
    state[index] = decayed;
    const float normalized_key =
        __half2float(raw_key[raw_offset + row]) * k_scale;
    memory += normalized_key * decayed;
  }

  const float delta =
      (value[vector_offset + column] - memory) * beta[bh];
  float result = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float normalized_key =
        __half2float(raw_key[raw_offset + row]) * k_scale;
    const float updated = state[index] + normalized_key * delta;
    state[index] = updated;
    const float normalized_query =
        __half2float(raw_query[raw_offset + row]) * q_scale;
    result += normalized_query * updated;
  }
  output[vector_offset + column] = result;
}

__global__ void gdn_mapped_recurrent_kernel(
    float* state, const half* normalized_query, const half* normalized_key,
    const float* value, const float* decay, const float* beta, float* output,
    int value_heads, int key_heads, int expand_ratio) {
  const int bh = blockIdx.x;
  const int column = threadIdx.x;
  const int batch = bh / value_heads;
  const int value_head = bh % value_heads;
  const int key_head = value_head / expand_ratio;
  const int key_offset = (batch * key_heads + key_head) * kHeadDim;
  const int vector_offset = bh * kHeadDim;
  const int state_offset = bh * kHeadDim * kHeadDim;
  const float head_decay = decay[bh];
  float memory = 0.0f;

#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float decayed = state[index] * head_decay;
    state[index] = decayed;
    memory += __half2float(normalized_key[key_offset + row]) * decayed;
  }

  const float delta =
      (value[vector_offset + column] - memory) * beta[bh];
  float result = 0.0f;
#pragma unroll
  for (int row = 0; row < kHeadDim; ++row) {
    const int index = state_offset + row * kHeadDim + column;
    const float key_value =
        __half2float(normalized_key[key_offset + row]);
    const float updated = state[index] + key_value * delta;
    state[index] = updated;
    const float query_value =
        __half2float(normalized_query[key_offset + row]) * 0.0883883476483f;
    result += query_value * updated;
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

void gdn_prep_recurrent_update_out(
    torch::Tensor state, const torch::Tensor& raw_query,
    const torch::Tensor& raw_key, const torch::Tensor& value,
    const torch::Tensor& decay, const torch::Tensor& beta,
    torch::Tensor output) {
  const int value_heads = static_cast<int>(state.size(1));
  const int key_heads = static_cast<int>(raw_query.size(1));
  const int blocks = static_cast<int>(state.size(0) * state.size(1));
  gdn_prep_recurrent_kernel<<<blocks, kHeadDim, 0,
                              at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(),
      reinterpret_cast<const half*>(raw_query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(raw_key.data_ptr<at::Half>()),
      value.data_ptr<float>(), decay.data_ptr<float>(), beta.data_ptr<float>(),
      output.data_ptr<float>(), value_heads, key_heads,
      value_heads / key_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gdn_mapped_recurrent_update_out(
    torch::Tensor state, const torch::Tensor& normalized_query,
    const torch::Tensor& normalized_key, const torch::Tensor& value,
    const torch::Tensor& decay, const torch::Tensor& beta,
    torch::Tensor output) {
  const int value_heads = static_cast<int>(state.size(1));
  const int key_heads = static_cast<int>(normalized_query.size(1));
  const int blocks = static_cast<int>(state.size(0) * state.size(1));
  gdn_mapped_recurrent_kernel<<<blocks, kHeadDim, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
      state.data_ptr<float>(),
      reinterpret_cast<const half*>(normalized_query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(normalized_key.data_ptr<at::Half>()),
      value.data_ptr<float>(), decay.data_ptr<float>(), beta.data_ptr<float>(),
      output.data_ptr<float>(), value_heads, key_heads,
      value_heads / key_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("recurrent_update", &gdn_recurrent_update,
             "Fused CoreX Gated DeltaNet recurrent update");
  module.def("recurrent_update_out", &gdn_recurrent_update_out,
             "Unchecked fused recurrent update with preallocated output");
  module.def("prep_recurrent_update_out", &gdn_prep_recurrent_update_out,
             "Fused q/k prep and recurrent update with preallocated output");
  module.def("mapped_recurrent_update_out", &gdn_mapped_recurrent_update_out,
             "Mapped normalized q/k recurrent update");
}
