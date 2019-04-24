import torch
from torch.autograd.function import InplaceFunction


# Assumed symmetric distribution of weights (i.e. range [-val, val])
class Quantize(InplaceFunction):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, w, numBits):
        # Bring to range [0, 1] reducing impact of large values
        w_q = torch.tanh(w).div(2 * torch.max(torch.abs(torch.tanh(w)))) + 0.5

        # Quantize to k bits in range [0, 1]
        w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)

        # Affine to bring to range [-1, 1]
        w_q *= 2
        w_q -= 1

        return w_q

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        import pdb
        pdb.set_trace()

        grad_input = None
        grad_input = grad_output

        return grad_input, None