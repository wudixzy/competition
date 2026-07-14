#include <torch/extension.h>

torch::Tensor gdn_recurrent_update_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor beta,
    torch::Tensor decay,
    torch::Tensor state);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "recurrent_update",
      &gdn_recurrent_update_cuda,
      "Fused BI100 GDN recurrent update");
}
