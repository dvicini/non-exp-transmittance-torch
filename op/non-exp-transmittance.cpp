#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
torch::Tensor tr_cuda_forward(torch::Tensor sigmat, torch::Tensor gamma);

std::vector<torch::Tensor> tr_cuda_backward(
    torch::Tensor grad_tr, torch::Tensor sigmat,
    torch::Tensor gamma, torch::Tensor tr_output);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor tr_forward(
    torch::Tensor sigmat,
    torch::Tensor gamma) {
  CHECK_INPUT(sigmat);
  CHECK_INPUT(gamma);
  return tr_cuda_forward(sigmat, gamma);
}

std::vector<torch::Tensor> tr_backward(
    torch::Tensor grad_tr,
    torch::Tensor sigmat,
    torch::Tensor gamma,
    torch::Tensor tr_output) {
  CHECK_INPUT(grad_tr);
  CHECK_INPUT(sigmat);
  CHECK_INPUT(gamma);
  CHECK_INPUT(tr_output);
  return tr_cuda_backward(grad_tr, sigmat, gamma, tr_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &tr_forward, "Non-exponential transmittance evaluation");
  m.def("backward", &tr_backward, "Non-exponential transmittance backward function");
}
