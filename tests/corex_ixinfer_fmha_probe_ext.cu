#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ixinfer.h>
#include <torch/extension.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

void check_cuinfer(cuinferStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUINFER_STATUS_SUCCESS, operation,
      " failed with cuinfer status ", static_cast<int>(status), " (",
      cuinferGetErrorString(status), ")");
}

void check_half_cuda_contiguous(const torch::Tensor& tensor,
                                const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 4, name, " must be four-dimensional");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(
      value >= 0 && value <= std::numeric_limits<int>::max(),
      name, " does not fit in int");
  return static_cast<int>(value);
}

struct CuinferHandle {
  cuinferHandle_t value = nullptr;

  CuinferHandle() {
    check_cuinfer(cuinferCreate(&value), "cuinferCreate");
  }

  ~CuinferHandle() {
    if (value != nullptr) {
      cuinferDestroy(value);
    }
  }

  CuinferHandle(const CuinferHandle&) = delete;
  CuinferHandle& operator=(const CuinferHandle&) = delete;
};

struct TensorDescriptor {
  cuinferTensorDescriptor_t value = nullptr;

  explicit TensorDescriptor(const torch::Tensor& tensor) {
    check_cuinfer(
        cuinferCreateTensorDescriptor(&value),
        "cuinferCreateTensorDescriptor");
    const std::array<int, 4> dimensions = {
        checked_int(tensor.size(0), "dimension 0"),
        checked_int(tensor.size(1), "dimension 1"),
        checked_int(tensor.size(2), "dimension 2"),
        checked_int(tensor.size(3), "dimension 3"),
    };
    const std::array<int, 4> strides = {
        checked_int(tensor.stride(0), "stride 0"),
        checked_int(tensor.stride(1), "stride 1"),
        checked_int(tensor.stride(2), "stride 2"),
        checked_int(tensor.stride(3), "stride 3"),
    };
    check_cuinfer(
        cuinferSetTensor4dDescriptorEx(
            value, CUINFER_DATA_HALF,
            dimensions[0], dimensions[1], dimensions[2], dimensions[3],
            strides[0], strides[1], strides[2], strides[3]),
        "cuinferSetTensor4dDescriptorEx");
  }

  ~TensorDescriptor() {
    if (value != nullptr) {
      cuinferDestroyTensorDescriptor(value);
    }
  }

  TensorDescriptor(const TensorDescriptor&) = delete;
  TensorDescriptor& operator=(const TensorDescriptor&) = delete;
};

}  // namespace

torch::Tensor ixinfer_fmha_forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    int64_t layout_arg) {
  check_half_cuda_contiguous(query, "query");
  check_half_cuda_contiguous(key, "key");
  check_half_cuda_contiguous(value, "value");
  TORCH_CHECK(
      query.device() == key.device() && query.device() == value.device(),
      "query, key, and value must use the same device");
  TORCH_CHECK(
      layout_arg == CUINFER_FATTN_BHSD
          || layout_arg == CUINFER_FATTN_BSHD,
      "layout must be 0 (BHSD) or 1 (BSHD)");

  const bool bshd = layout_arg == CUINFER_FATTN_BSHD;
  const int sequence_dimension = bshd ? 1 : 2;
  const int head_dimension = bshd ? 2 : 1;
  const int64_t batch = query.size(0);
  const int64_t query_length = query.size(sequence_dimension);
  const int64_t key_length = key.size(sequence_dimension);
  const int64_t query_heads = query.size(head_dimension);
  const int64_t kv_heads = key.size(head_dimension);
  const int64_t head_size = query.size(3);

  TORCH_CHECK(batch == 1, "the isolated probe supports batch size one");
  TORCH_CHECK(key.size(0) == batch && value.size(0) == batch,
              "batch sizes must match");
  TORCH_CHECK(key.sizes() == value.sizes(),
              "key and value shapes must match");
  TORCH_CHECK(query_length > 0 && key_length >= query_length,
              "key length must be at least query length");
  TORCH_CHECK(query_heads > 0 && kv_heads > 0,
              "head counts must be positive");
  TORCH_CHECK(query_heads % kv_heads == 0,
              "query head count must be divisible by KV head count");
  TORCH_CHECK(key.size(3) == head_size && value.size(3) == head_size,
              "head sizes must match");
  TORCH_CHECK(
      head_size == 128 || head_size == 256,
      "M1-160 supports head size 128 as a capability control and "
      "production head size 256");

  c10::cuda::CUDAGuard device_guard(query.device());
  auto output = torch::empty_like(query);
  TensorDescriptor query_descriptor(query);
  TensorDescriptor key_descriptor(key);
  TensorDescriptor value_descriptor(value);
  TensorDescriptor output_descriptor(output);
  CuinferHandle handle;
  check_cuinfer(
      cuinferSetStream(handle.value, at::cuda::getCurrentCUDAStream()),
      "cuinferSetStream");

  cuinferFlashAttnConfigInfo config{};
  config.layout =
      static_cast<cuinferFlashAttnLayout_t>(layout_arg);
  config.isCausal = causal;
  config.scaling = 1.0f / std::sqrt(static_cast<float>(head_size));
  config.qoSeqArray = nullptr;
  config.kvSeqArray = nullptr;
  config.kvSeqStart = 0;
  config.kvSeqEnd = checked_int(key_length, "key length");
  config.kvHeadNum = checked_int(kv_heads, "KV head count");
  config.isAlibi = false;
  config.alibiMode = CUINFER_FATTN_ALIBI_MODE_SUB_KQ;
  config.slopeM = nullptr;
  config.qStride = checked_int(
      query.stride(sequence_dimension), "query token stride");
  config.kStride = checked_int(
      key.stride(sequence_dimension), "key token stride");
  config.vStride = checked_int(
      value.stride(sequence_dimension), "value token stride");

  check_cuinfer(
      cuinferFMHAForwardEx(
          handle.value, config,
          query_descriptor.value, query.data_ptr<at::Half>(),
          key_descriptor.value, key.data_ptr<at::Half>(),
          value_descriptor.value, value.data_ptr<at::Half>(),
          nullptr, nullptr,
          output_descriptor.value, output.data_ptr<at::Half>()),
      "cuinferFMHAForwardEx");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "forward", &ixinfer_fmha_forward,
      "Isolated CoreX ixinfer FMHAForwardEx capability probe");
}
