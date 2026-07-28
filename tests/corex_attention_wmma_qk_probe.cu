#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>

using namespace nvcuda;

namespace {

constexpr int kM = 16;
constexpr int kN = 32;
constexpr int kK = 32;
constexpr int kHead = 256;
constexpr int kWarp = 64;

__global__ void qk_kernel(
    const half* query, const half* key, float* output, int64_t tiles) {
  const int64_t tile = blockIdx.x;
  const int lane = threadIdx.x;
  if (tile >= tiles || lane >= kWarp) {
    return;
  }

  __shared__ half query_shared[kM * kK];
  __shared__ half key_shared[kK * kN];
  __shared__ float output_shared[kM * kN];
  wmma::fragment<wmma::accumulator, kM, kN, kK, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);

  for (int k_tile = 0; k_tile < kHead / kK; ++k_tile) {
    for (int index = 0; index < 4; ++index) {
      const int row = (lane / 16) * 2 + 8 * (index % 2);
      const int column = lane % 16 + 16 * (index / 2);
      const int offset =
          wmma::CoordToOffset<16, wmma::layout_t::mem_row_major>(
              row, column);
      half2 value;
      value.x = query[(tile * kM + row) * kHead
                      + k_tile * kK + column];
      value.y = query[(tile * kM + row + 1) * kHead
                      + k_tile * kK + column];
      *reinterpret_cast<int*>(&query_shared[offset]) = __HALF2_TO_UI(value);
    }

    for (int key_half = 0; key_half < 2; ++key_half) {
      for (int index = 0; index < 4; ++index) {
        const int row = (lane / 16) * 2 + 8 * (index % 2);
        const int column = lane % 16 + 16 * (index / 2);
        const int offset =
            wmma::CoordToOffset<16, wmma::layout_t::mem_row_major>(
                row, column);
        const int key_dim = key_half * 16 + row;
        half2 value;
        value.x = key[(tile * kN + column) * kHead
                      + k_tile * kK + key_dim];
        value.y = key[(tile * kN + column) * kHead
                      + k_tile * kK + key_dim + 1];
        *reinterpret_cast<int*>(
            &key_shared[key_half * 16 * kN + offset]) =
            __HALF2_TO_UI(value);
      }
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, kM, kN, kK,
                   half, wmma::row_major> query_fragment;
    wmma::fragment<wmma::matrix_b, kM, kN, kK,
                   half, wmma::row_major> key_fragment;
    wmma::load_matrix_sync(query_fragment, query_shared, 0);
    wmma::load_matrix_sync(key_fragment, key_shared, 0);
    wmma::mma_sync(
        accumulator, query_fragment, key_fragment, accumulator);
    __syncthreads();
  }

  wmma::store_matrix_sync(
      output_shared, accumulator, 0, wmma::mem_row_major);
  __syncthreads();
  for (int n_half = 0; n_half < 2; ++n_half) {
    for (int row_group = 0; row_group < 4; ++row_group) {
      const int row = lane / 16 + 4 * row_group;
      const int column = 16 * n_half + lane % 16;
      const int offset =
          wmma::CoordToOffset<32, wmma::layout_t::mem_row_major>(
              row, lane % 16);
      output[(tile * kM + row) * kN + column] =
          output_shared[n_half * 16 * 16 + offset];
    }
  }
}

torch::Tensor qk(const torch::Tensor& query, const torch::Tensor& key) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(),
              "query and key must be CUDA tensors");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16
                  && key.scalar_type() == torch::kFloat16,
              "query and key must have dtype float16");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(),
              "query and key must be contiguous");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3,
              "query and key must be rank three");
  TORCH_CHECK(query.size(1) == kM && query.size(2) == kHead,
              "query must have shape [tiles, 16, 256]");
  TORCH_CHECK(key.size(0) == query.size(0)
                  && key.size(1) == kN && key.size(2) == kHead,
              "key must have shape [tiles, 32, 256]");
  TORCH_CHECK(query.device() == key.device(),
              "query and key must share one device");

  auto output = torch::empty(
      {query.size(0), kM, kN},
      query.options().dtype(torch::kFloat32));
  qk_kernel<<<query.size(0), kWarp, 0,
              at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key.data_ptr<at::Half>()),
      output.data_ptr<float>(), query.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("qk", &qk, "BI100 WMMA 16x32 QK capability probe");
}
