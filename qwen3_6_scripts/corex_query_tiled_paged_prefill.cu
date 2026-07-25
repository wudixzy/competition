#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
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
constexpr int kNumQueryHeads = 4;
constexpr int kNumKvHeads = 1;
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 16;
constexpr int kMmaK = 16;
constexpr int kDimTiles = kHeadDim / kMmaK;
constexpr int kWarpSize = 64;
constexpr int kMaxQueryTokens = 8192;
constexpr int kMaxSequenceTokens = 262144;

using namespace nvcuda;

struct __align__(128) SharedStorage {
  float matrix_tile[kQueryTile * kKeyTile];
  float scores[kQueryTile * kKeyTile];
  float running_output[kQueryTile * kHeadDim];
  float tile_output[kQueryTile * kMmaK];
  float running_max[kQueryTile];
  float running_sum[kQueryTile];
  float correction[kQueryTile];
};

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ float load_key(
    const __half* key_new, const __half* key_cache,
    const int* block_table, int logical_token, int context_len, int dim) {
  if (logical_token < context_len) {
    const int logical_block = logical_token / kBlockSize;
    const int block_offset = logical_token % kBlockSize;
    const int physical_block = block_table[logical_block];
    const int64_t index =
        ((((static_cast<int64_t>(physical_block) * kNumKvHeads)
            * (kHeadDim / kKeyPack) + dim / kKeyPack)
           * kBlockSize + block_offset) * kKeyPack + dim % kKeyPack);
    return __half2float(key_cache[index]);
  }
  const int query_index = logical_token - context_len;
  return __half2float(
      key_new[static_cast<int64_t>(query_index) * kHeadDim + dim]);
}

__device__ __forceinline__ float load_value(
    const __half* value_new, const __half* value_cache,
    const int* block_table, int logical_token, int context_len, int dim) {
  if (logical_token < context_len) {
    const int logical_block = logical_token / kBlockSize;
    const int block_offset = logical_token % kBlockSize;
    const int physical_block = block_table[logical_block];
    const int64_t index =
        ((static_cast<int64_t>(physical_block) * kNumKvHeads)
          * kHeadDim + dim) * kBlockSize + block_offset;
    return __half2float(value_cache[index]);
  }
  const int query_index = logical_token - context_len;
  return __half2float(
      value_new[static_cast<int64_t>(query_index) * kHeadDim + dim]);
}

__global__ void query_tiled_paged_prefill_kernel(
    const __half* query, const __half* key_new, const __half* value_new,
    const __half* key_cache, const __half* value_cache,
    const int* block_table, __half* output, float* lse,
    int context_len, int query_len, float scale) {
  __shared__ SharedStorage shared;

  const int lane = threadIdx.x;
  const int query_tile_index = blockIdx.x / kNumQueryHeads;
  const int query_head = blockIdx.x % kNumQueryHeads;
  const int query_start = query_tile_index * kQueryTile;
  const int active_rows = min(kQueryTile, query_len - query_start);
  if (active_rows <= 0) {
    return;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, float,
                 wmma::row_major> query_fragments[kDimTiles];

#pragma unroll
  for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter) {
      const int row = lane / 16 + quarter * 4;
      const int column = lane % 16;
      float value = 0.0f;
      if (row < active_rows) {
        const int query_index = query_start + row;
        const int dim = dim_tile * kMmaK + column;
        const int64_t source =
            (static_cast<int64_t>(query_index) * kNumQueryHeads
              + query_head) * kHeadDim + dim;
        value = __half2float(query[source]) * scale;
      }
      const int offset =
          wmma::CoordToOffset<32, wmma::layout_t::mem_row_major>(
              row, column);
      shared.matrix_tile[offset] = value;
    }
    __syncthreads();
    wmma::load_matrix_sync(
        query_fragments[dim_tile], shared.matrix_tile, 0);
    __syncthreads();
  }

  for (int index = lane; index < kQueryTile * kHeadDim;
       index += kWarpSize) {
    shared.running_output[index] = 0.0f;
  }
  if (lane < kQueryTile) {
    shared.running_max[lane] = -std::numeric_limits<float>::infinity();
    shared.running_sum[lane] = 0.0f;
    shared.correction[lane] = 1.0f;
  }
  __syncthreads();

  const int last_query = min(query_start + kQueryTile, query_len);
  const int visible_key_tokens = context_len + last_query;
  const int key_tiles =
      (visible_key_tokens + kKeyTile - 1) / kKeyTile;

  for (int key_tile_index = 0; key_tile_index < key_tiles;
       ++key_tile_index) {
    const int key_start = key_tile_index * kKeyTile;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
    wmma::fill_fragment(score_fragment, 0.0f);

#pragma unroll
    for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        const int row = lane / 16 + quarter * 4;
        const int column = lane % 16;
        const int logical_token = key_start + column;
        const int dim = dim_tile * kMmaK + row;
        const float value =
            logical_token < visible_key_tokens
                ? load_key(key_new, key_cache, block_table,
                           logical_token, context_len, dim)
                : 0.0f;
        const int offset =
            wmma::CoordToOffset<32, wmma::layout_t::mem_col_major>(
                row, column);
        shared.matrix_tile[offset] = value;
      }
      __syncthreads();
      wmma::fragment<wmma::matrix_b, 16, 16, 16, float,
                     wmma::col_major> key_fragment;
      wmma::load_matrix_sync(key_fragment, shared.matrix_tile, 0);
      wmma::mma_sync(
          score_fragment,
          query_fragments[dim_tile],
          key_fragment,
          score_fragment);
      __syncthreads();
    }

    wmma::store_matrix_sync(
        shared.scores, score_fragment, 0, wmma::mem_row_major);
    __syncthreads();

    if (lane < kQueryTile) {
      const int row = lane;
      if (row >= active_rows) {
        shared.correction[row] = 1.0f;
#pragma unroll
        for (int column = 0; column < kKeyTile; ++column) {
          shared.scores[row * kKeyTile + column] = 0.0f;
        }
      } else {
        const int absolute_query = context_len + query_start + row;
        float block_max = -std::numeric_limits<float>::infinity();
#pragma unroll
        for (int column = 0; column < kKeyTile; ++column) {
          const int logical_key = key_start + column;
          if (logical_key <= absolute_query
              && logical_key < visible_key_tokens) {
            block_max = fmaxf(
                block_max, shared.scores[row * kKeyTile + column]);
          } else {
            shared.scores[row * kKeyTile + column] =
                -std::numeric_limits<float>::infinity();
          }
        }
        const float old_max = shared.running_max[row];
        const float new_max = fmaxf(old_max, block_max);
        const float correction =
            old_max == -std::numeric_limits<float>::infinity()
                ? 0.0f
                : expf(old_max - new_max);
        float tile_sum = 0.0f;
#pragma unroll
        for (int column = 0; column < kKeyTile; ++column) {
          const float score = shared.scores[row * kKeyTile + column];
          const float probability =
              score == -std::numeric_limits<float>::infinity()
                  ? 0.0f
                  : expf(score - new_max);
          shared.scores[row * kKeyTile + column] = probability;
          tile_sum += probability;
        }
        shared.running_sum[row] =
            shared.running_sum[row] * correction + tile_sum;
        shared.running_max[row] = new_max;
        shared.correction[row] = correction;
      }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, float,
                   wmma::row_major> probability_fragment;
    wmma::load_matrix_sync(
        probability_fragment, shared.scores, 0);

#pragma unroll
    for (int dim_tile = 0; dim_tile < kDimTiles; ++dim_tile) {
#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        const int row = lane / 16 + quarter * 4;
        const int column = lane % 16;
        const int logical_token = key_start + row;
        const int dim = dim_tile * kMmaK + column;
        const float value =
            logical_token < visible_key_tokens
                ? load_value(value_new, value_cache, block_table,
                             logical_token, context_len, dim)
                : 0.0f;
        const int offset =
            wmma::CoordToOffset<32, wmma::layout_t::mem_row_major>(
                row, column);
        shared.matrix_tile[offset] = value;
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_b, 16, 16, 16, float,
                     wmma::row_major> value_fragment;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float>
          output_fragment;
      wmma::load_matrix_sync(
          value_fragment, shared.matrix_tile, 0);
      wmma::fill_fragment(output_fragment, 0.0f);
      wmma::mma_sync(
          output_fragment,
          probability_fragment,
          value_fragment,
          output_fragment);
      wmma::store_matrix_sync(
          shared.tile_output, output_fragment, 0, wmma::mem_row_major);
      __syncthreads();

#pragma unroll
      for (int quarter = 0; quarter < 4; ++quarter) {
        const int row = lane / 16 + quarter * 4;
        const int column = lane % 16;
        if (row < active_rows) {
          const int output_index =
              row * kHeadDim + dim_tile * kMmaK + column;
          const int tile_index = row * kMmaK + column;
          shared.running_output[output_index] =
              shared.running_output[output_index]
                  * shared.correction[row]
              + shared.tile_output[tile_index];
        }
      }
      __syncthreads();
    }
  }

  for (int index = lane; index < active_rows * kHeadDim;
       index += kWarpSize) {
    const int row = index / kHeadDim;
    const int dim = index % kHeadDim;
    const int query_index = query_start + row;
    const int64_t destination =
        (static_cast<int64_t>(query_index) * kNumQueryHeads
          + query_head) * kHeadDim + dim;
    output[destination] = __float2half_rn(
        shared.running_output[index] / shared.running_sum[row]);
  }
  if (lane < active_rows) {
    const int query_index = query_start + lane;
    lse[static_cast<int64_t>(query_index) * kNumQueryHeads
        + query_head] =
        shared.running_max[lane] + logf(shared.running_sum[lane]);
  }
}

}  // namespace

std::vector<torch::Tensor> query_tiled_paged_prefill_forward(
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
              "query must have shape (Q, 4, 256)");
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
  const int required_blocks = context_len / kBlockSize;
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

  auto output = torch::empty_like(query);
  auto lse = torch::empty(
      {query_len, kNumQueryHeads},
      query.options().dtype(torch::kFloat32));
  const int query_tiles =
      (query_len + kQueryTile - 1) / kQueryTile;
  const int blocks = query_tiles * kNumQueryHeads;
  auto stream = at::cuda::getCurrentCUDAStream();
  query_tiled_paged_prefill_kernel<<<
      blocks, kWarpSize, 0, stream>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_new.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_new.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      lse.data_ptr<float>(), context_len, query_len,
      static_cast<float>(scale_arg));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, lse};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &query_tiled_paged_prefill_forward,
             "Fixed BI100 query-tiled paged-prefill forward");
}
