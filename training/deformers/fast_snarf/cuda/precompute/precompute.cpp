#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>





void launch_precompute(const torch::Tensor &voxel_w, const torch::Tensor &tfs, torch::Tensor &voxel_d, torch::Tensor &voxel_J, const torch::Tensor &offset, const torch::Tensor &scale);

void precompute(const torch::Tensor &voxel_w, const torch::Tensor &tfs, torch::Tensor &voxel_d, torch::Tensor &voxel_J, const torch::Tensor &offset, const torch::Tensor &scale) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(voxel_w));

  launch_precompute(voxel_w, tfs, voxel_d, voxel_J, offset, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("precompute", &precompute);
}
