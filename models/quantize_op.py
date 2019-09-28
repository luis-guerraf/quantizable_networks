import torch
from torch.autograd import Function

class DoReFa_A(Function):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    @staticmethod
    def forward(ctx, x, numBits):
        ctx.save_for_backward(x)
        # Bring to range [0, 1] reducing impact of large values
        w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

        # Quantize to k bits in range [0, 1]
        w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)

        # Affine to bring to range [-1, 1]
        w_q *= 2
        w_q -= 1

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output, None


class DoReFa_W(Function):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    @staticmethod
    def forward(ctx, x, numBits):
        # Bring to range [0, 1] reducing impact of large values
        w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

        # Quantize to k bits in range [0, 1]
        w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)

        # Affine to bring to range [-1, 1]
        w_q *= 2
        w_q -= 1

        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
