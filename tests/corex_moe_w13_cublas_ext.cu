#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor linear_cublas(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            int64_t mode) {
  check_half_cuda(input, "input");
  check_half_cuda(weight, "weight");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1,
              "input must have shape (1, input_size)");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == input.size(1),
              "weight must have shape (output_size, input_size)");
  TORCH_CHECK(mode == -2 || mode == -1
                  || (mode >= 0 && mode <= 23)
                  || (mode >= 99 && mode <= 115),
              "unsupported cuBLAS probe mode");

  const int output_size = static_cast<int>(weight.size(0));
  const int input_size = static_cast<int>(weight.size(1));
  auto output = torch::empty({1, output_size}, input.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CHECK(cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
              "cublasSetStream failed");

  cublasStatus_t status;
  if (mode == -2) {
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    status = cublasHgemm(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        output_size, 1, input_size,
        &alpha,
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        input_size,
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        input_size,
        &beta,
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        output_size);
  } else {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        output_size, 1, input_size,
        &alpha,
        weight.data_ptr<at::Half>(), CUDA_R_16F, input_size,
        input.data_ptr<at::Half>(), CUDA_R_16F, input_size,
        &beta,
        output.data_ptr<at::Half>(), CUDA_R_16F, output_size,
        CUDA_R_32F, static_cast<cublasGemmAlgo_t>(mode));
  }
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "cuBLAS linear mode ", mode, " failed with status ",
              static_cast<int>(status));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("linear", &linear_cublas,
             "Contiguous FP16 W13 linear cuBLAS algorithm probe");
}
