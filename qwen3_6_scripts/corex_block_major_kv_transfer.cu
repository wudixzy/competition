#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxGridBlocks = 65535;

__global__ void pack_layer_kernel(
    const __half* source, const int64_t* block_ids, __half* staging,
    int64_t row_elements, int rows, int layer, int layers,
    int source_blocks) {
  const int64_t total = static_cast<int64_t>(rows) * row_elements;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(index / row_elements);
    const int64_t within_row = index % row_elements;
    const int kv = static_cast<int>(within_row / (row_elements / 2));
    const int64_t payload_index = within_row % (row_elements / 2);
    const int64_t source_index =
        ((static_cast<int64_t>(kv) * source_blocks + block_ids[row])
         * (row_elements / 2)) + payload_index;
    const int64_t staging_index =
        ((static_cast<int64_t>(row) * layers + layer) * row_elements)
        + within_row;
    staging[staging_index] = source[source_index];
  }
}

__global__ void scatter_layer_kernel(
    const __half* staging, const int64_t* block_ids, __half* destination,
    int64_t row_elements, int rows, int layer, int layers,
    int destination_blocks) {
  const int64_t total = static_cast<int64_t>(rows) * row_elements;
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int row = static_cast<int>(index / row_elements);
    const int64_t within_row = index % row_elements;
    const int kv = static_cast<int>(within_row / (row_elements / 2));
    const int64_t payload_index = within_row % (row_elements / 2);
    const int64_t staging_index =
        ((static_cast<int64_t>(row) * layers + layer) * row_elements)
        + within_row;
    const int64_t destination_index =
        ((static_cast<int64_t>(kv) * destination_blocks + block_ids[row])
         * (row_elements / 2)) + payload_index;
    destination[destination_index] = staging[staging_index];
  }
}

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t validate_inputs(const std::vector<torch::Tensor>& gpu_cache,
                        const torch::Tensor& block_ids,
                        const torch::Tensor& staging) {
  TORCH_CHECK(!gpu_cache.empty(), "gpu_cache must contain at least one layer");
  TORCH_CHECK(block_ids.is_cuda(), "block_ids must be a CUDA tensor");
  TORCH_CHECK(block_ids.scalar_type() == torch::kInt64,
              "block_ids must have dtype int64");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be one-dimensional");
  check_half_cuda_contiguous(staging, "staging");
  TORCH_CHECK(staging.dim() == 4,
              "staging must have shape [rows, layers, 2, payload]");
  TORCH_CHECK(staging.size(0) == block_ids.numel(),
              "staging row count must match block_ids");
  TORCH_CHECK(staging.size(1) == static_cast<int64_t>(gpu_cache.size()),
              "staging layer count must match gpu_cache");
  TORCH_CHECK(staging.size(2) == 2, "staging K/V dimension must equal 2");
  TORCH_CHECK(staging.size(3) > 0, "staging payload must be non-empty");

  int64_t block_count = -1;
  const int64_t payload = staging.size(3);
  for (size_t layer = 0; layer < gpu_cache.size(); ++layer) {
    const auto& cache = gpu_cache[layer];
    check_half_cuda_contiguous(cache, "gpu_cache layer");
    TORCH_CHECK(cache.device() == staging.device(),
                "all cache layers and staging must share one device");
    TORCH_CHECK(cache.dim() >= 3 && cache.size(0) == 2,
                "cache layer must have shape [2, blocks, ...]");
    TORCH_CHECK(cache.size(1) > 0, "cache layer has no blocks");
    TORCH_CHECK(cache.numel() == 2 * cache.size(1) * payload,
                "cache layer payload differs from staging");
    if (block_count < 0) {
      block_count = cache.size(1);
    } else {
      TORCH_CHECK(cache.size(1) == block_count,
                  "cache layer block counts differ");
    }
  }
  if (block_ids.numel() > 0) {
    const int64_t minimum = block_ids.min().item<int64_t>();
    const int64_t maximum = block_ids.max().item<int64_t>();
    TORCH_CHECK(minimum >= 0 && maximum < block_count,
                "block_ids must be within [0, ", block_count, ")");
  }
  return block_count;
}

int launch_blocks(int64_t total) {
  return static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kMaxGridBlocks));
}

}  // namespace

void pack_block_major(const std::vector<torch::Tensor>& gpu_cache,
                      const torch::Tensor& block_ids,
                      const torch::Tensor& staging) {
  const int64_t block_count = validate_inputs(gpu_cache, block_ids, staging);
  if (block_ids.numel() == 0) {
    return;
  }
  const int rows = static_cast<int>(block_ids.numel());
  const int layers = static_cast<int>(gpu_cache.size());
  const int64_t row_elements = 2 * staging.size(3);
  const int blocks = launch_blocks(static_cast<int64_t>(rows) * row_elements);
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (int layer = 0; layer < layers; ++layer) {
    pack_layer_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(
            gpu_cache[layer].data_ptr<at::Half>()),
        block_ids.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(staging.data_ptr<at::Half>()),
        row_elements, rows, layer, layers, static_cast<int>(block_count));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scatter_block_major(const torch::Tensor& staging,
                         const std::vector<torch::Tensor>& gpu_cache,
                         const torch::Tensor& block_ids) {
  const int64_t block_count = validate_inputs(gpu_cache, block_ids, staging);
  if (block_ids.numel() == 0) {
    return;
  }
  const int rows = static_cast<int>(block_ids.numel());
  const int layers = static_cast<int>(gpu_cache.size());
  const int64_t row_elements = 2 * staging.size(3);
  const int blocks = launch_blocks(static_cast<int64_t>(rows) * row_elements);
  const auto stream = at::cuda::getCurrentCUDAStream();
  for (int layer = 0; layer < layers; ++layer) {
    scatter_layer_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const __half*>(staging.data_ptr<at::Half>()),
        block_ids.data_ptr<int64_t>(),
        reinterpret_cast<__half*>(
            gpu_cache[layer].data_ptr<at::Half>()),
        row_elements, rows, layer, layers, static_cast<int>(block_count));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("pack", &pack_block_major,
             "Pack layer-major FP16 KV blocks into block-major staging");
  module.def("scatter", &scatter_block_major,
             "Scatter block-major FP16 staging into layer-major KV blocks");
}
