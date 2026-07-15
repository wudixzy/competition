#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <vector>

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kThreads = 256;

void check_half_matrix(const torch::Tensor& input, const char* name) {
  TORCH_CHECK(input.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(input.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHiddenSize,
              name, " must have shape (rows, 2048)");
}

__global__ void prepare_kernel(const __half* input, float* converted,
                               float* squares, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const float value = __half2float(input[index]);
    converted[index] = value;
    squares[index] = __fmul_rn(value, value);
  }
}

__global__ void prepare_residual_kernel(
    const __half* input, const __half* residual, __half* summed,
    float* converted, float* squares, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x
                       + threadIdx.x;
       index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const __half sum = __float2half_rn(
        __half2float(input[index]) + __half2float(residual[index]));
    const float value = __half2float(sum);
    summed[index] = sum;
    converted[index] = value;
    squares[index] = __fmul_rn(value, value);
  }
}

__global__ void apply_inverse_kernel(
    const float* input, const __half* weight, const float* inverse,
    __half* output, int rows) {
  const int row = blockIdx.x;
  for (int column = threadIdx.x; column < kHiddenSize;
       column += blockDim.x) {
    const int offset = row * kHiddenSize + column;
    const float scaled = __fmul_rn(input[offset], inverse[row]);
    const float factor = __fadd_rn(1.0f, __half2float(weight[column]));
    output[offset] = __float2half_rn(__fmul_rn(scaled, factor));
  }
}

int grid_for(int64_t elements) {
  return static_cast<int>((elements + kThreads - 1) / kThreads);
}

}  // namespace

std::vector<torch::Tensor> prepare(const torch::Tensor& input) {
  check_half_matrix(input, "input");
  auto float_options = input.options().dtype(torch::kFloat32);
  auto converted = torch::empty(input.sizes(), float_options);
  auto squares = torch::empty(input.sizes(), float_options);
  const int64_t elements = input.numel();
  prepare_kernel<<<grid_for(elements), kThreads, 0,
                   at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      converted.data_ptr<float>(), squares.data_ptr<float>(), elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {converted, squares};
}

std::vector<torch::Tensor> prepare_residual(
    const torch::Tensor& input, const torch::Tensor& residual) {
  check_half_matrix(input, "input");
  check_half_matrix(residual, "residual");
  TORCH_CHECK(residual.sizes() == input.sizes(),
              "residual must match input shape");
  auto float_options = input.options().dtype(torch::kFloat32);
  auto converted = torch::empty(input.sizes(), float_options);
  auto squares = torch::empty(input.sizes(), float_options);
  auto summed = torch::empty_like(input);
  const int64_t elements = input.numel();
  prepare_residual_kernel<<<grid_for(elements), kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(summed.data_ptr<at::Half>()),
      converted.data_ptr<float>(), squares.data_ptr<float>(), elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {converted, squares, summed};
}

torch::Tensor apply_inverse(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            const torch::Tensor& inverse) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda() && inverse.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32,
              "input must have dtype float32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16,
              "weight must have dtype float16");
  TORCH_CHECK(inverse.scalar_type() == torch::kFloat32,
              "inverse must have dtype float32");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous()
                  && inverse.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHiddenSize,
              "input must have shape (rows, 2048)");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == kHiddenSize,
              "weight must have shape (2048,)");
  TORCH_CHECK(inverse.numel() == input.size(0),
              "inverse must contain one value per row");
  auto output = torch::empty(
      input.sizes(), input.options().dtype(torch::kFloat16));
  apply_inverse_kernel<<<input.size(0), kThreads, 0,
                         at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<float>(),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      inverse.data_ptr<float>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      static_cast<int>(input.size(0)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("prepare", &prepare,
             "Convert FP16 Gemma RMSNorm input and compute exact squares");
  module.def("prepare_residual", &prepare_residual,
             "Add residual, convert, and compute exact Gemma RMSNorm squares");
  module.def("apply_inverse", &apply_inverse,
             "Apply PyTorch-computed Gemma RMSNorm inverse exactly");
}
