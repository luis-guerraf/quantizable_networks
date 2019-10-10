import torch
from torch.autograd import Function

def DoReFa_A(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    # w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def DoReFa_W(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    # w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


class RoundNoGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g
