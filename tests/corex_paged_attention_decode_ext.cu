#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kPartitionTokens = 512;

__device__ __forceinline__ int64_t key_offset(
    int physical_block, int kv_head, int dim, int block_offset,
    int num_kv_heads, int head_size, int block_size, int key_pack) {
  return (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head)
           * (head_size / key_pack) + dim / key_pack)
          * block_size + block_offset) * key_pack + dim % key_pack;
}

__device__ __forceinline__ int64_t value_offset(
    int physical_block, int kv_head, int dim, int block_offset,
    int num_kv_heads, int head_size, int block_size) {
  return ((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head)
          * head_size + dim) * block_size + block_offset;
}

__global__ void paged_attention_partitions_kernel(
    const __half* query, const __half* key_cache, const __half* value_cache,
    const int* block_table, float* partial_output, float* partial_sums,
    float* partial_maxes, int seq_len, int num_heads, int num_kv_heads,
    int head_size, int block_size, int key_pack, int num_partitions,
    float scale) {
  __shared__ float scores[kPartitionTokens];
  __shared__ float reduction[kThreads];
  __shared__ float scaled_query[kThreads];

  const int head = blockIdx.x;
  const int partition = blockIdx.y;
  const int tid = threadIdx.x;
  const int kv_head = head / (num_heads / num_kv_heads);
  const int partition_start = partition * kPartitionTokens;
  const int partition_len = min(kPartitionTokens, seq_len - partition_start);

  scaled_query[tid] = __half2float(
      query[static_cast<int64_t>(head) * head_size + tid]) * scale;
  __syncthreads();

#pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int local_token = tid + item * kThreads;
    float score = -INFINITY;
    if (local_token < partition_len) {
      const int token = partition_start + local_token;
      const int logical_block = token / block_size;
      const int block_offset = token % block_size;
      const int physical_block = block_table[logical_block];
      score = 0.0f;
#pragma unroll 4
      for (int dim = 0; dim < kThreads; ++dim) {
        score += scaled_query[dim] * __half2float(key_cache[key_offset(
            physical_block, kv_head, dim, block_offset, num_kv_heads,
            head_size, block_size, key_pack)]);
      }
    }
    scores[local_token] = score;
  }
  __syncthreads();

  reduction[tid] = fmaxf(scores[tid], scores[tid + kThreads]);
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float partition_max = reduction[0];

  float local_sum = 0.0f;
#pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int local_token = tid + item * kThreads;
    const float weight = (local_token < partition_len)
        ? expf(scores[local_token] - partition_max) : 0.0f;
    scores[local_token] = weight;
    local_sum += weight;
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float partition_sum = reduction[0];

  float weighted_value = 0.0f;
  for (int local_token = 0; local_token < partition_len; ++local_token) {
    const int token = partition_start + local_token;
    const int logical_block = token / block_size;
    const int block_offset = token % block_size;
    const int physical_block = block_table[logical_block];
    weighted_value += scores[local_token] * __half2float(
        value_cache[value_offset(physical_block, kv_head, tid, block_offset,
                                 num_kv_heads, head_size, block_size)]);
  }

  const int64_t partition_index =
      static_cast<int64_t>(head) * num_partitions + partition;
  partial_output[partition_index * head_size + tid] = weighted_value;
  if (tid == 0) {
    partial_sums[partition_index] = partition_sum;
    partial_maxes[partition_index] = partition_max;
  }
}

__global__ void paged_attention_reduce_kernel(
    const float* partial_output, const float* partial_sums,
    const float* partial_maxes, __half* output, int num_partitions,
    int head_size) {
  __shared__ float reduction[kThreads];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int64_t base = static_cast<int64_t>(head) * num_partitions;

  float local_max = -INFINITY;
  for (int partition = tid; partition < num_partitions;
       partition += kThreads) {
    local_max = fmaxf(local_max, partial_maxes[base + partition]);
  }
  reduction[tid] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
    }
    __syncthreads();
  }
  const float global_max = reduction[0];

  float local_sum = 0.0f;
  for (int partition = tid; partition < num_partitions;
       partition += kThreads) {
    const float correction = expf(
        partial_maxes[base + partition] - global_max);
    local_sum += correction * partial_sums[base + partition];
  }
  reduction[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      reduction[tid] += reduction[tid + stride];
    }
    __syncthreads();
  }
  const float global_sum = reduction[0];

  float value = 0.0f;
  for (int partition = 0; partition < num_partitions; ++partition) {
    const float correction = expf(
        partial_maxes[base + partition] - global_max);
    value += correction * partial_output[
        (base + partition) * head_size + tid];
  }
  output[static_cast<int64_t>(head) * head_size + tid] =
      __float2half_rn(value / global_sum);
}

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor paged_attention_decode(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, const torch::Tensor& block_table,
    int64_t seq_len, double scale) {
  check_half_cuda_contiguous(query, "query");
  check_half_cuda_contiguous(key_cache, "key_cache");
  check_half_cuda_contiguous(value_cache, "value_cache");
  TORCH_CHECK(block_table.is_cuda() && block_table.is_contiguous(),
              "block_table must be a contiguous CUDA tensor");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32,
              "block_table must have dtype int32");
  TORCH_CHECK(query.dim() == 3 && query.size(0) == 1,
              "query must have shape (1, heads, head_size)");
  TORCH_CHECK(key_cache.dim() == 5 && value_cache.dim() == 4,
              "unexpected paged cache rank");
  TORCH_CHECK(block_table.dim() == 1,
              "block_table must be a one-dimensional row");
  TORCH_CHECK(query.size(2) == kThreads,
              "prototype requires head_size=256");
  TORCH_CHECK(key_cache.size(0) == value_cache.size(0)
                  && key_cache.size(1) == value_cache.size(1)
                  && key_cache.size(3) == value_cache.size(3)
                  && key_cache.size(2) * key_cache.size(4) == query.size(2)
                  && value_cache.size(2) == query.size(2),
              "inconsistent query and cache shapes");
  TORCH_CHECK(query.size(1) % key_cache.size(1) == 0,
              "query heads must be divisible by KV heads");
  TORCH_CHECK(seq_len > 0, "seq_len must be positive");
  const int block_size = static_cast<int>(value_cache.size(3));
  TORCH_CHECK((seq_len + block_size - 1) / block_size <= block_table.numel(),
              "block_table is too short for seq_len");

  const int num_heads = static_cast<int>(query.size(1));
  const int num_kv_heads = static_cast<int>(key_cache.size(1));
  const int head_size = static_cast<int>(query.size(2));
  const int key_pack = static_cast<int>(key_cache.size(4));
  const int num_partitions = static_cast<int>(
      (seq_len + kPartitionTokens - 1) / kPartitionTokens);
  auto float_options = query.options().dtype(torch::kFloat32);
  auto partial_output = torch::empty(
      {num_heads, num_partitions, head_size}, float_options);
  auto partial_sums = torch::empty(
      {num_heads, num_partitions}, float_options);
  auto partial_maxes = torch::empty_like(partial_sums);
  auto output = torch::empty_like(query);

  const dim3 partition_grid(num_heads, num_partitions);
  paged_attention_partitions_kernel<<<
      partition_grid, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(value_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int>(), partial_output.data_ptr<float>(),
      partial_sums.data_ptr<float>(), partial_maxes.data_ptr<float>(),
      static_cast<int>(seq_len), num_heads, num_kv_heads, head_size,
      block_size, key_pack, num_partitions, static_cast<float>(scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  paged_attention_reduce_kernel<<<
      num_heads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      partial_output.data_ptr<float>(), partial_sums.data_ptr<float>(),
      partial_maxes.data_ptr<float>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      num_partitions, head_size);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &paged_attention_decode,
             "Split-K direct paged attention decode prototype");
}
