#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__global__ void w13_serial_matvec_kernel(
    const __half* input, const __half* weight, __half* output,
    int rows, int columns) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }
  float sum = 0.0f;
  const __half* row_weight = weight + static_cast<int64_t>(row) * columns;
  for (int column = 0; column < columns; ++column) {
    sum = fmaf(__half2float(row_weight[column]),
               __half2float(input[column]), sum);
  }
  output[row] = __float2half_rn(sum);
}

__global__ void w13_warp_matvec_kernel(
    const __half* input, const __half* weight, __half* output,
    int rows, int columns) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= rows) {
    return;
  }
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* row2 = reinterpret_cast<const __half2*>(
      weight + static_cast<int64_t>(warp) * columns);
  float sum = 0.0f;
  for (int index = lane; index < columns / 2; index += kWarpSize) {
    const __half2 w = row2[index];
    const __half2 x = input2[index];
    sum = fmaf(__half2float(w.x), __half2float(x.x), sum);
    sum = fmaf(__half2float(w.y), __half2float(x.y), sum);
  }
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) {
    output[warp] = __float2half_rn(sum);
  }
}

template <int kThreadsPerRow>
__global__ void w13_segmented_matvec_kernel(
    const __half* input, const __half* weight, __half* output,
    int rows, int columns) {
  __shared__ float partials[kThreads];
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int local = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  float sum = 0.0f;
  if (row < rows) {
    const __half2* input2 = reinterpret_cast<const __half2*>(input);
    const __half2* row2 = reinterpret_cast<const __half2*>(
        weight + static_cast<int64_t>(row) * columns);
    for (int index = local; index < columns / 2;
         index += kThreadsPerRow) {
      const __half2 w = row2[index];
      const __half2 x = input2[index];
      sum = fmaf(__half2float(w.x), __half2float(x.x), sum);
      sum = fmaf(__half2float(w.y), __half2float(x.y), sum);
    }
  }
  partials[threadIdx.x] = sum;
  __syncthreads();
  if (local == 0 && row < rows) {
    float total = partials[threadIdx.x];
#pragma unroll
    for (int item = 1; item < kThreadsPerRow; ++item) {
      total += partials[threadIdx.x + item];
    }
    output[row] = __float2half_rn(total);
  }
}

template <int kThreadsPerRow>
__global__ void w13_segmented_double_matvec_kernel(
    const __half* input, const __half* weight, __half* output,
    int rows, int columns) {
  __shared__ double partials[kThreads];
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int local = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  double sum = 0.0;
  if (row < rows) {
    const __half2* input2 = reinterpret_cast<const __half2*>(input);
    const __half2* row2 = reinterpret_cast<const __half2*>(
        weight + static_cast<int64_t>(row) * columns);
    for (int index = local; index < columns / 2;
         index += kThreadsPerRow) {
      const __half2 w = row2[index];
      const __half2 x = input2[index];
      sum = fma(static_cast<double>(__half2float(w.x)),
                static_cast<double>(__half2float(x.x)), sum);
      sum = fma(static_cast<double>(__half2float(w.y)),
                static_cast<double>(__half2float(x.y)), sum);
    }
  }
  partials[threadIdx.x] = sum;
  __syncthreads();
  if (local == 0 && row < rows) {
    double total = partials[threadIdx.x];
#pragma unroll
    for (int item = 1; item < kThreadsPerRow; ++item) {
      total += partials[threadIdx.x + item];
    }
    output[row] = __float2half_rn(static_cast<float>(total));
  }
}

__device__ inline void kahan_add(float value, float& sum, float& correction) {
  const float adjusted = value - correction;
  const float updated = sum + adjusted;
  correction = (updated - sum) - adjusted;
  sum = updated;
}

template <int kThreadsPerRow>
__global__ void w13_segmented_kahan_matvec_kernel(
    const __half* input, const __half* weight, __half* output,
    int rows, int columns) {
  __shared__ float partials[kThreads];
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int local = threadIdx.x % kThreadsPerRow;
  const int row_in_block = threadIdx.x / kThreadsPerRow;
  const int row = blockIdx.x * kRowsPerBlock + row_in_block;
  float sum = 0.0f;
  float correction = 0.0f;
  if (row < rows) {
    const __half2* input2 = reinterpret_cast<const __half2*>(input);
    const __half2* row2 = reinterpret_cast<const __half2*>(
        weight + static_cast<int64_t>(row) * columns);
    for (int index = local; index < columns / 2;
         index += kThreadsPerRow) {
      const __half2 w = row2[index];
      const __half2 x = input2[index];
      kahan_add(__half2float(w.x) * __half2float(x.x), sum, correction);
      kahan_add(__half2float(w.y) * __half2float(x.y), sum, correction);
    }
  }
  partials[threadIdx.x] = sum;
  __syncthreads();
  if (local == 0 && row < rows) {
    float total = 0.0f;
    float total_correction = 0.0f;
#pragma unroll
    for (int item = 0; item < kThreadsPerRow; ++item) {
      kahan_add(partials[threadIdx.x + item], total, total_correction);
    }
    output[row] = __float2half_rn(total);
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

template <int kThreadsPerRow>
void launch_segmented(const torch::Tensor& input,
                      const torch::Tensor& weight,
                      torch::Tensor& output, int rows, int columns) {
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  w13_segmented_matvec_kernel<kThreadsPerRow><<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      rows, columns);
}

template <int kThreadsPerRow>
void launch_segmented_double(const torch::Tensor& input,
                             const torch::Tensor& weight,
                             torch::Tensor& output,
                             int rows, int columns) {
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  w13_segmented_double_matvec_kernel<kThreadsPerRow><<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      rows, columns);
}

template <int kThreadsPerRow>
void launch_segmented_kahan(const torch::Tensor& input,
                            const torch::Tensor& weight,
                            torch::Tensor& output,
                            int rows, int columns) {
  constexpr int kRowsPerBlock = kThreads / kThreadsPerRow;
  const int blocks = (rows + kRowsPerBlock - 1) / kRowsPerBlock;
  w13_segmented_kahan_matvec_kernel<kThreadsPerRow><<<
      blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      rows, columns);
}

}  // namespace

torch::Tensor linear_w13_matvec(const torch::Tensor& input,
                                const torch::Tensor& weight,
                                int64_t mode) {
  check_half_cuda(input, "input");
  check_half_cuda(weight, "weight");
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1,
              "input must have shape (1, input_size)");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == input.size(1),
              "weight must have shape (output_size, input_size)");
  TORCH_CHECK(weight.size(1) % 2 == 0,
              "input size must be divisible by two");
  TORCH_CHECK(mode == 0 || mode == 1 || mode == 32 || mode == 64
                  || mode == 128 || mode == 256
                  || mode == 1032 || mode == 1064
                  || mode == 1128 || mode == 1256
                  || mode == 2032 || mode == 2064
                  || mode == 2128 || mode == 2256,
              "unsupported W13 matvec mode");

  const int rows = static_cast<int>(weight.size(0));
  const int columns = static_cast<int>(weight.size(1));
  auto output = torch::empty({1, rows}, input.options());
  if (mode == 0) {
    const int warps_per_block = kThreads / kWarpSize;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    w13_warp_matvec_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows, columns);
  } else if (mode == 1) {
    const int blocks = (rows + kThreads - 1) / kThreads;
    w13_serial_matvec_kernel<<<
        blocks, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows, columns);
  } else if (mode == 32) {
    launch_segmented<32>(input, weight, output, rows, columns);
  } else if (mode == 64) {
    launch_segmented<64>(input, weight, output, rows, columns);
  } else if (mode == 128) {
    launch_segmented<128>(input, weight, output, rows, columns);
  } else if (mode == 256) {
    launch_segmented<256>(input, weight, output, rows, columns);
  } else if (mode == 1032) {
    launch_segmented_double<32>(input, weight, output, rows, columns);
  } else if (mode == 1064) {
    launch_segmented_double<64>(input, weight, output, rows, columns);
  } else if (mode == 1128) {
    launch_segmented_double<128>(input, weight, output, rows, columns);
  } else if (mode == 1256) {
    launch_segmented_double<256>(input, weight, output, rows, columns);
  } else if (mode == 2032) {
    launch_segmented_kahan<32>(input, weight, output, rows, columns);
  } else if (mode == 2064) {
    launch_segmented_kahan<64>(input, weight, output, rows, columns);
  } else if (mode == 2128) {
    launch_segmented_kahan<128>(input, weight, output, rows, columns);
  } else {
    launch_segmented_kahan<256>(input, weight, output, rows, columns);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("linear", &linear_w13_matvec,
             "Shape-specific FP16 W13 matrix-vector probe");
}
