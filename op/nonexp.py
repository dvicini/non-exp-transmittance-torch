import torch

import _nonexp_cuda

class Transmittance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigmat, gamma):
        outputs = _nonexp_cuda.forward(sigmat, gamma)
        ctx.save_for_backward(sigmat, gamma, outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_tr):
        grad = _nonexp_cuda.backward(grad_tr.contiguous(), *ctx.saved_tensors)
        return grad[0], grad[1]
