#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

template <typename Float>
__device__ __forceinline__ Float clamp(Float v, Float a, Float b) {
    return max(a, min(b, v));
}

template <typename Float>
__device__ __forceinline__ Float f(Float tau, Float gamma) {
    return gamma * exp(-tau) + (1 - gamma) * max(1 - 0.5f * tau, 0.f);
}

template <typename Float>
__device__ __forceinline__ Float f_d_tau(Float tau, Float gamma) {
    return -gamma * exp(-tau) - (1 - gamma) * (1 - 0.5f * tau > 0.0f) * 0.5f;
}

template <typename Float>
__device__ __forceinline__ Float f_d_gamma(Float tau, Float gamma) {
    return exp(-tau) - max(1 - 0.5f * tau, 0.0f);
}

template <typename Float>
__device__ __forceinline__ Float f_inv(Float y, Float gamma) {
    // Inverse of the transmittance function "f" wrt. the first argument

    y = clamp(y, Float(0), Float(1));
    const size_t max_iter = 5;
    Float x = (gamma < 0.1f) ? 1 - y : -log(max(y, 0.0001f));
    for (size_t i = 0; i < max_iter; ++i) {
        Float deriv = f_d_tau(x, gamma);
        if (deriv == 0.f)
            deriv = 1.f;
        x = x - (f(x, gamma) - y) / deriv;
        x = max(x, 0.f);
    }
    return x;
}

template <typename Float>
__device__ __forceinline__ std::pair<Float, Float> f_inv_d(Float y, Float gamma) {
    // Computes the derivatives of f_inv using the inverse function theorem
    Float tau = f_inv(y, gamma);
    Float d_tr = 1 / f_d_tau(tau, gamma);
    Float d_gamma = -f_d_gamma(tau, gamma) * d_tr;
    if ((d_tr != d_tr) || (d_gamma != d_gamma) || (y == 0.0)) {
        d_tr = 0.f;
        d_gamma = 0.f;
    }

    // Make sure the derivatives don't explode
    d_tr = clamp(d_tr, Float(-100.f), Float(100.f));
    d_gamma = clamp(d_gamma, Float(-2.f), Float(2.f));
    return std::make_pair(d_tr, d_gamma);
}

template <typename Float>
__global__ void tr_cuda_forward_kernel(
        const torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> sigmat,
        const torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> gamma,
        torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> tr_output) {

    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= sigmat.size(0))
        return;

    Float tr_value = 1.f;
    for (size_t i = 0; i < sigmat.size(1); ++i) {
        Float gamma_v = gamma[ray][i];
        Float tr_new = min(tr_value, f(f_inv(tr_value, gamma_v) + sigmat[ray][i], gamma_v));
        tr_output[ray][i] = tr_new; // store output transmittance
        tr_value = tr_new;
    }
}

template <typename Float>
__global__ void tr_cuda_backward_kernel(
        torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> grad_tr,
        const torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> sigmat,
        const torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> gamma,
        const torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> tr_output,
        torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> d_sigmat,
        torch::PackedTensorAccessor<Float,2,torch::RestrictPtrTraits,size_t> d_gamma) {

    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= sigmat.size(0))
        return;

    Float tr = tr_output[ray][sigmat.size(1) - 1];
    Float tr_jac = 0.0f;
    for (int i = sigmat.size(1) - 1; i >= 0; --i) {

        Float gamma_v = gamma[ray][i];
        Float sigmat_v = sigmat[ray][i];

        Float prev_tr = tr_output[ray][max(i - 1, 0)];
        // In the paper, we use inverse computation to not need to store tr_output. However, since this op
        // outputs the transmittance for each step, there is little benefit in doing that. It
        // is easier to just access the already stored values.
        // Float prev_tr = f(f_inv(tr, gamma_v) - sigmat_v, gamma_v);
        if (i == 0)
            prev_tr = 1.0f;

        Float tr_optical_depth = f_inv(prev_tr, gamma_v) + sigmat_v;
        auto derivs = f_inv_d(prev_tr, gamma_v);
        Float f_inv_d_sigma = derivs.first;
        Float f_inv_d_gamma = derivs.second;

        Float sigmat_grad = f_d_tau(tr_optical_depth, gamma_v);
        Float gamma_grad = f_d_tau(tr_optical_depth, gamma_v) * f_inv_d_gamma;
        gamma_grad += f_d_gamma(tr_optical_depth, gamma_v);

        // Exploit linearity of the backpropagation to reduce complexity to linear
        tr_jac += grad_tr[ray][i];

        d_sigmat[ray][i] += sigmat_grad * tr_jac;
        d_gamma[ray][i] += gamma_grad * tr_jac;
        tr = prev_tr;
        tr_jac *= (f_d_tau(tr_optical_depth, gamma_v) * f_inv_d_sigma);
    }
}

} // namespace

torch::Tensor tr_cuda_forward(
    torch::Tensor sigmat,
    torch::Tensor gamma) {

    const auto n_rays = sigmat.size(0);
    const auto n_samples = sigmat.size(1);
    auto tr_output = torch::zeros_like(sigmat);

    size_t block_size = 1024;
    size_t n_blocks = (n_rays + block_size - 1) / block_size;
    block_size = min(block_size, n_rays);
    AT_DISPATCH_FLOATING_TYPES(sigmat.type(), "tr_forward_cuda", ([&] {
        tr_cuda_forward_kernel<scalar_t><<<n_blocks, block_size>>>(
            sigmat.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            gamma.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            tr_output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
    }));
    return tr_output;
}

std::vector<torch::Tensor> tr_cuda_backward(
    torch::Tensor grad_tr,
    torch::Tensor sigmat,
    torch::Tensor gamma,
    torch::Tensor tr_output) {

    auto d_sigmat = torch::zeros_like(sigmat);
    auto d_gamma = torch::zeros_like(gamma);
    const auto n_rays = sigmat.size(0);

    size_t block_size = 1024;
    size_t n_blocks = (n_rays + block_size - 1) / block_size;
    block_size = min(block_size, n_rays);
    AT_DISPATCH_FLOATING_TYPES(sigmat.type(), "tr_backward_cuda", ([&] {
        tr_cuda_backward_kernel<scalar_t><<<n_blocks, block_size>>>(
            grad_tr.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            sigmat.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            gamma.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            tr_output.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            d_sigmat.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            d_gamma.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>());
        }));
  return {d_sigmat, d_gamma};
}
