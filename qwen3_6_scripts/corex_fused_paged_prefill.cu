#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kBlockSize = 16;
constexpr int kHeadDim = 256;
constexpr int kKeyPack = 8;
constexpr int kNumQueryHeads = 6;
constexpr int kNumKvHeads = 1;
constexpr int kTileTokens = 512;
constexpr int kThreads = 256;
constexpr int kMaxQueryTokens = 8192;
constexpr int kMaxSequenceTokens = 262144;

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__global__ void convert_query_kernel(const __half* query, float* converted,
                                     int query_len, float scale) {
  const int64_t total = static_cast<int64_t>(query_len)
      * kNumQueryHeads * kHeadDim;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int dim = index % kHeadDim;
    const int query_index =
        (index / kHeadDim) % query_len;
    const int head =
        index / (static_cast<int64_t>(kHeadDim) * query_len);
    const int64_t source =
        (static_cast<int64_t>(query_index) * kNumQueryHeads + head)
        * kHeadDim + dim;
    converted[index] = __half2float(query[source]) * scale;
  }
}

__global__ void gather_kv_tile_kernel(
    const __half* key_new, const __half* value_new,
    const __half* key_cache, const __half* value_cache,
    const int* block_table, float* key_tile, float* value_tile,
    int context_len, int query_len, int tile_start, int valid_tokens) {
  constexpr int kElements = kTileTokens * kHeadDim;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < kElements; index += blockDim.x * gridDim.x) {
    const int token_offset = index / kHeadDim;
    const int dim = index - token_offset * kHeadDim;
    const int logical_token = tile_start + token_offset;
    float key_value = 0.0f;
    float value_value = 0.0f;
    if (token_offset >= valid_tokens) {
      // The fixed 512-column GEMMs require zero-filled tail columns.
    } else if (logical_token < context_len) {
      const int logical_block = logical_token / kBlockSize;
      const int block_offset = logical_token % kBlockSize;
      const int physical_block = block_table[logical_block];
      const int64_t key_index =
          (((static_cast<int64_t>(physical_block) * kNumKvHeads)
             * (kHeadDim / kKeyPack) + dim / kKeyPack)
            * kBlockSize + block_offset) * kKeyPack + dim % kKeyPack;
      const int64_t value_index =
          ((static_cast<int64_t>(physical_block) * kNumKvHeads)
             * kHeadDim + dim) * kBlockSize + block_offset;
      key_value = __half2float(key_cache[key_index]);
      value_value = __half2float(value_cache[value_index]);
    } else if (logical_token < context_len + query_len) {
      const int query_index = logical_token - context_len;
      const int64_t source =
          static_cast<int64_t>(query_index) * kHeadDim + dim;
      key_value = __half2float(key_new[source]);
      value_value = __half2float(value_new[source]);
    }
    key_tile[index] = key_value;
    value_tile[index] = value_value;
  }
}

__global__ void mask_causal_scores_kernel(
    float* scores, int query_len, int context_len,
    int tile_start, int valid_tokens, int rows) {
  const int64_t elements =
      static_cast<int64_t>(rows) * valid_tokens;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int column = index % valid_tokens;
    const int row = index / valid_tokens;
    const int query_index = row % query_len;
    if (tile_start + column > context_len + query_index) {
      scores[static_cast<int64_t>(row) * kTileTokens + column] =
          -std::numeric_limits<float>::infinity();
    }
  }
}

__global__ void accumulate_output_kernel(
    float* running_output, const float* tile_output,
    const float* correction, int64_t elements) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = index / kHeadDim;
    const float scaled =
        __fmul_rn(running_output[index], correction[row]);
    running_output[index] = __fadd_rn(scaled, tile_output[index]);
  }
}

int launch_blocks(int64_t elements) {
  const int64_t needed = (elements + kThreads - 1) / kThreads;
  return static_cast<int>(std::min<int64_t>(needed, 65535));
}

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

cublasStatus_t qk_batched(
    cublasHandle_t handle, const float* key_tile, const float* query,
    float* scores, int query_len) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasSgemmStridedBatched(
      handle, CUBLAS_OP_T, CUBLAS_OP_N,
      kTileTokens, query_len, kHeadDim,
      &alpha, key_tile, kHeadDim, 0,
      query, kHeadDim, static_cast<long long>(query_len) * kHeadDim,
      &beta, scores, kTileTokens,
      static_cast<long long>(query_len) * kTileTokens,
      kNumQueryHeads);
}

cublasStatus_t pv_batched(
    cublasHandle_t handle, const float* value_tile, const float* scores,
    float* output, int query_len) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      kHeadDim, query_len, kTileTokens,
      &alpha, value_tile, kHeadDim, 0,
      scores, kTileTokens,
      static_cast<long long>(query_len) * kTileTokens,
      &beta, output, kHeadDim,
      static_cast<long long>(query_len) * kHeadDim,
      kNumQueryHeads);
}

}  // namespace

std::vector<torch::Tensor> fused_paged_prefill_forward(
    const torch::Tensor& query, const torch::Tensor& key_new,
    const torch::Tensor& value_new, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_table,
    int64_t context_len_arg, double scale_arg) {
  check_half_cuda_contiguous(query, "query");
  check_half_cuda_contiguous(key_new, "key_new");
  check_half_cuda_contiguous(value_new, "value_new");
  check_half_cuda_contiguous(key_cache, "key_cache");
  check_half_cuda_contiguous(value_cache, "value_cache");
  TORCH_CHECK(block_table.is_cuda(),
              "block_table must be a CUDA tensor");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must have dtype int32");
  TORCH_CHECK(block_table.is_contiguous(),
              "block_table must be contiguous");
  TORCH_CHECK(block_table.dim() == 1,
              "block_table must be one-dimensional");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == kNumQueryHeads
                  && query.size(2) == kHeadDim,
              "query must have shape (Q, 6, 256)");
  TORCH_CHECK(key_new.dim() == 3 && key_new.size(1) == kNumKvHeads
                  && key_new.size(2) == kHeadDim,
              "key_new must have shape (Q, 1, 256)");
  TORCH_CHECK(value_new.sizes() == key_new.sizes(),
              "value_new must match key_new");
  TORCH_CHECK(key_new.size(0) == query.size(0),
              "query, key_new, and value_new lengths must match");
  TORCH_CHECK(key_cache.dim() == 5
                  && key_cache.size(1) == kNumKvHeads
                  && key_cache.size(2) == kHeadDim / kKeyPack
                  && key_cache.size(3) == kBlockSize
                  && key_cache.size(4) == kKeyPack,
              "key_cache must have shape (N, 1, 32, 16, 8)");
  TORCH_CHECK(value_cache.dim() == 4
                  && value_cache.size(1) == kNumKvHeads
                  && value_cache.size(2) == kHeadDim
                  && value_cache.size(3) == kBlockSize,
              "value_cache must have shape (N, 1, 256, 16)");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0),
              "key/value cache block counts must match");
  TORCH_CHECK(query.device() == key_new.device()
                  && query.device() == value_new.device()
                  && query.device() == key_cache.device()
                  && query.device() == value_cache.device()
                  && query.device() == block_table.device(),
              "all tensors must use the same device");
  TORCH_CHECK(context_len_arg >= 0
                  && context_len_arg <= kMaxSequenceTokens,
              "context_len is out of range");
  TORCH_CHECK(context_len_arg % kBlockSize == 0,
              "context_len must be block aligned");
  const int query_len = static_cast<int>(query.size(0));
  const int context_len = static_cast<int>(context_len_arg);
  TORCH_CHECK(query_len > 0 && query_len <= kMaxQueryTokens,
              "query length must be in [1, 8192]");
  TORCH_CHECK(context_len + query_len <= kMaxSequenceTokens,
              "context_len + query_len exceeds 262144");
  const int required_blocks =
      (context_len + kBlockSize - 1) / kBlockSize;
  TORCH_CHECK(block_table.numel() >= required_blocks,
              "block_table is too short for context_len");
  if (required_blocks > 0) {
    auto active_blocks = block_table.narrow(0, 0, required_blocks);
    const int minimum_block = active_blocks.min().item<int>();
    const int maximum_block = active_blocks.max().item<int>();
    TORCH_CHECK(minimum_block >= 0
                    && maximum_block < key_cache.size(0),
                "block_table contains an out-of-range physical block ID");
  }
  TORCH_CHECK(std::isfinite(scale_arg) && scale_arg > 0.0,
              "scale must be finite and positive");
  TORCH_CHECK(query_len <= std::numeric_limits<int>::max() / kNumQueryHeads,
              "query length overflows row count");

  const int rows = kNumQueryHeads * query_len;
  const int64_t output_elements =
      static_cast<int64_t>(rows) * kHeadDim;
  auto float_options = query.options().dtype(torch::kFloat32);
  auto converted_query = torch::empty(
      {kNumQueryHeads, query_len, kHeadDim}, float_options);
  auto key_tile = torch::empty({kTileTokens, kHeadDim}, float_options);
  auto value_tile = torch::empty({kTileTokens, kHeadDim}, float_options);
  auto scores = torch::empty(
      {kNumQueryHeads, query_len, kTileTokens}, float_options);
  auto tile_output = torch::empty(
      {kNumQueryHeads, query_len, kHeadDim}, float_options);
  auto running_max = torch::full(
      {kNumQueryHeads, query_len},
      -std::numeric_limits<float>::infinity(), float_options);
  auto running_sum = torch::zeros(
      {kNumQueryHeads, query_len}, float_options);
  auto running_output = torch::zeros(
      {kNumQueryHeads, query_len, kHeadDim}, float_options);

  auto stream = at::cuda::getCurrentCUDAStream();
  convert_query_kernel<<<launch_blocks(output_elements), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      converted_query.data_ptr<float>(), query_len,
      static_cast<float>(scale_arg));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
  const auto run_tile = [&](int tile_start, int valid_tokens) {
    constexpr int kGatherBlocks = 512;
    gather_kv_tile_kernel<<<kGatherBlocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(key_new.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_new.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(), key_tile.data_ptr<float>(),
        value_tile.data_ptr<float>(), context_len, query_len, tile_start,
        valid_tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    check_cublas(qk_batched(
        handle, key_tile.data_ptr<float>(),
        converted_query.data_ptr<float>(), scores.data_ptr<float>(),
        query_len), "paged prefill QK");
    const int64_t active_score_elements =
        static_cast<int64_t>(rows) * valid_tokens;
    mask_causal_scores_kernel<<<
        launch_blocks(active_score_elements), kThreads, 0, stream>>>(
        scores.data_ptr<float>(), query_len, context_len, tile_start,
        valid_tokens, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto active_scores = scores.narrow(2, 0, valid_tokens);
    auto block_max = std::get<0>(at::max(active_scores, -1, false));
    auto new_max = at::maximum(running_max, block_max);
    active_scores.sub_(new_max.unsqueeze(-1)).exp_();
    auto correction = at::exp(running_max - new_max);
    running_max.copy_(new_max);
    running_sum.mul_(correction).add_(
        at::sum(active_scores, {-1}, false));

    check_cublas(pv_batched(
        handle, value_tile.data_ptr<float>(), scores.data_ptr<float>(),
        tile_output.data_ptr<float>(), query_len), "paged prefill PV");
    accumulate_output_kernel<<<
        launch_blocks(output_elements), kThreads, 0, stream>>>(
        running_output.data_ptr<float>(), tile_output.data_ptr<float>(),
        correction.data_ptr<float>(), output_elements);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  for (int tile_start = 0; tile_start < context_len;
       tile_start += kTileTokens) {
    run_tile(tile_start, std::min(kTileTokens, context_len - tile_start));
  }
  for (int key_start = 0; key_start < query_len;
       key_start += kTileTokens) {
    run_tile(context_len + key_start,
             std::min(kTileTokens, query_len - key_start));
  }

  running_output.div_(running_sum.unsqueeze(-1));
  auto output = running_output.permute({1, 0, 2})
                    .to(query.scalar_type()).contiguous();
  auto lse = (running_max + at::log(running_sum))
                 .transpose(0, 1).contiguous();
  return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &fused_paged_prefill_forward,
             "Fixed-shape FP32 paged-prefill pipeline for cache-only context");
}
