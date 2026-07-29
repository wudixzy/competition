#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kTileTokens = 512;
constexpr int kMaxQueryTokens = 8192;

void check_shape(const torch::Tensor& query,
                 const torch::Tensor& key) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda(),
              "query and key must be CUDA tensors");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous(),
              "query and key must be contiguous");
  TORCH_CHECK(query.device() == key.device(),
              "query and key must use the same device");
  TORCH_CHECK(query.dim() == 3 && query.size(0) == kHeads
                  && query.size(2) == kHeadDim,
              "query must have shape (4, Q, 256)");
  TORCH_CHECK(query.size(1) > 0 && query.size(1) <= kMaxQueryTokens,
              "query length must be in [1, 8192]");
  TORCH_CHECK(key.dim() == 2 && key.size(0) == kTileTokens
                  && key.size(1) == kHeadDim,
              "key must have shape (512, 256)");
}

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, operation,
              " failed with cuBLAS status ", static_cast<int>(status));
}

cublasHandle_t current_handle() {
  auto stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
  return handle;
}

}  // namespace

torch::Tensor qk_sgemm(const torch::Tensor& scaled_query,
                       const torch::Tensor& key) {
  check_shape(scaled_query, key);
  TORCH_CHECK(scaled_query.scalar_type() == torch::kFloat32
                  && key.scalar_type() == torch::kFloat32,
              "SGEMM query and key must have dtype float32");
  const int query_tokens = static_cast<int>(scaled_query.size(1));
  auto output = torch::empty(
      {kHeads, query_tokens, kTileTokens}, scaled_query.options());
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
      cublasSgemmStridedBatched(
          current_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
          kTileTokens, query_tokens, kHeadDim,
          &alpha, key.data_ptr<float>(), kHeadDim, 0,
          scaled_query.data_ptr<float>(), kHeadDim,
          static_cast<long long>(query_tokens) * kHeadDim,
          &beta, output.data_ptr<float>(), kTileTokens,
          static_cast<long long>(query_tokens) * kTileTokens,
          kHeads),
      "FP32 strided-batched QK");
  return output;
}

torch::Tensor qk_half_input(const torch::Tensor& query,
                            const torch::Tensor& key,
                            double scale_arg) {
  check_shape(query, key);
  TORCH_CHECK(query.scalar_type() == torch::kFloat16
                  && key.scalar_type() == torch::kFloat16,
              "GemmEx query and key must have dtype float16");
  TORCH_CHECK(std::isfinite(scale_arg) && scale_arg > 0.0,
              "scale must be finite and positive");
  const int query_tokens = static_cast<int>(query.size(1));
  auto output = torch::empty(
      {kHeads, query_tokens, kTileTokens},
      query.options().dtype(torch::kFloat32));
  const float alpha = static_cast<float>(scale_arg);
  const float beta = 0.0f;
  check_cublas(
      cublasGemmStridedBatchedEx(
          current_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
          kTileTokens, query_tokens, kHeadDim,
          &alpha,
          key.data_ptr<at::Half>(), CUDA_R_16F, kHeadDim, 0,
          query.data_ptr<at::Half>(), CUDA_R_16F, kHeadDim,
          static_cast<long long>(query_tokens) * kHeadDim,
          &beta,
          output.data_ptr<float>(), CUDA_R_32F, kTileTokens,
          static_cast<long long>(query_tokens) * kTileTokens,
          kHeads, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "FP16-input FP32-accumulate strided-batched QK");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("qk_sgemm", &qk_sgemm,
             "Production-shape FP32 strided-batched QK control");
  module.def("qk_half_input", &qk_half_input,
             "Production-shape FP16-input FP32-accumulate QK candidate");
}
