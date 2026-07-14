#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kTopK = 8;

__global__ void build_pointer_arrays(
    const half* weights, const half* input, half* output,
    const int64_t* expert_ids, unsigned long long* workspace,
    int input_size, int output_size, bool batched_input) {
  const int index = threadIdx.x;
  if (index >= kTopK) {
    return;
  }
  auto** weight_ptrs = reinterpret_cast<const half**>(workspace);
  auto** input_ptrs = reinterpret_cast<const half**>(workspace + kTopK);
  auto** output_ptrs = reinterpret_cast<half**>(workspace + 2 * kTopK);
  weight_ptrs[index] =
      weights + expert_ids[index] * output_size * input_size;
  input_ptrs[index] = input + (batched_input ? index * input_size : 0);
  output_ptrs[index] = output + index * output_size;
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void selected_gemv_out(const torch::Tensor& input,
                       const torch::Tensor& expert_ids,
                       const torch::Tensor& weights,
                       torch::Tensor output,
                       torch::Tensor pointer_workspace,
                       bool batched_input,
                       bool fp32_accumulate) {
  check_half_cuda(input, "input");
  check_half_cuda(weights, "weights");
  check_half_cuda(output, "output");
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.scalar_type() == torch::kInt64
                  && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA int64 tensor");
  TORCH_CHECK(pointer_workspace.is_cuda()
                  && pointer_workspace.scalar_type() == torch::kInt64
                  && pointer_workspace.is_contiguous()
                  && pointer_workspace.numel() >= 3 * kTopK,
              "pointer_workspace must contain at least 24 CUDA int64 values");
  TORCH_CHECK(weights.dim() == 3 && weights.size(0) >= kTopK,
              "weights must have shape (experts, output, input)");
  TORCH_CHECK(expert_ids.numel() == kTopK,
              "this probe requires exactly eight selected experts");

  const int output_size = static_cast<int>(weights.size(1));
  const int input_size = static_cast<int>(weights.size(2));
  TORCH_CHECK(output.numel() == kTopK * output_size,
              "output has the wrong number of elements");
  TORCH_CHECK(input.numel() ==
                  (batched_input ? kTopK * input_size : input_size),
              "input has the wrong number of elements");

  auto stream = at::cuda::getCurrentCUDAStream();
  build_pointer_arrays<<<1, 32, 0, stream>>>(
      reinterpret_cast<const half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<unsigned long long*>(
          pointer_workspace.data_ptr<int64_t>()),
      input_size, output_size, batched_input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto* workspace = reinterpret_cast<unsigned long long*>(
      pointer_workspace.data_ptr<int64_t>());
  auto** weight_ptrs = reinterpret_cast<const void**>(workspace);
  auto** input_ptrs = reinterpret_cast<const void**>(workspace + kTopK);
  auto** output_ptrs = reinterpret_cast<void**>(workspace + 2 * kTopK);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
              "cublasSetStream failed");

  cublasStatus_t status;
  if (fp32_accumulate) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cublasGemmBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, output_size, 1, input_size,
        &alpha, weight_ptrs, CUDA_R_16F, input_size,
        input_ptrs, CUDA_R_16F, input_size, &beta,
        output_ptrs, CUDA_R_16F, output_size, kTopK,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  } else {
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    status = cublasHgemmBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, output_size, 1, input_size,
        &alpha, reinterpret_cast<const half* const*>(weight_ptrs), input_size,
        reinterpret_cast<const half* const*>(input_ptrs), input_size, &beta,
        reinterpret_cast<half* const*>(output_ptrs), output_size, kTopK);
  }
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "batched selected GEMV failed with cuBLAS status ",
              static_cast<int>(status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("selected_gemv_out", &selected_gemv_out,
             "No-copy selected-expert batched GEMV");
}
