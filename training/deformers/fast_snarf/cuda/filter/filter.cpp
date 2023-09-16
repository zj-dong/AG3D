#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>





void launch_filter(
  torch::Tensor &output, const torch::Tensor &x, const torch::Tensor &mask);

void filter(const torch::Tensor &x, const torch::Tensor &mask, torch::Tensor &output) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  launch_filter(output, x, mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("filter", &filter);
}
