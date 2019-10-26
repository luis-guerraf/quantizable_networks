import torch
from torch.autograd import Function
import math

class Binarize_A(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input > 1] = 0
        grad_output[input < -1] = 0
        return grad_output


class Binarize_W(Function):
    @staticmethod
    def forward(ctx, x):
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def TanhQuant_A(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])
    if numBits == 32:
        # Just bring to range [-1, 1]
        return torch.tanh(x).div(torch.max(torch.abs(torch.tanh(x))))

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def DoReFa_A(x, numBits):
    # DoReFa is the same than Half-wave Gaussian (uniform) Quantizer. They clip to [0,1]
    w_q = torch.clamp(x, min=0.0, max=1.0)

    if numBits == 32:
        return x

    # Quantize to k bits in range [0, 1]
    w_q = w_q.mul(2 ** numBits - 1)
    w_q = RoundNoGradient.apply(w_q)
    w_q = w_q.div(2 ** numBits - 1)

    return w_q


# NOTE: Modifying this gradients is meaningless, because these are the gradients of self.weight.org
# not self.weight. To supress the gradients in self.weight do it directly in the gradients vector of
# self.weight
def DoReFa_W(x, numBits):
    # Assumed symmetric distribution of weights (i.e. range [-val, val])

    if numBits == 32:
        # Just bring to range [-1, 1]
        return torch.tanh(x).div(torch.max(torch.abs(torch.tanh(x))))

    # Bring to range [0, 1] reducing impact of large values
    w_q = torch.tanh(x).div(2 * torch.max(torch.abs(torch.tanh(x)))) + 0.5

    # Quantize to k bits in range [0, 1]
    w_q = w_q.mul(2 ** numBits - 1).round().div(2 ** numBits - 1)

    # Affine to bring to range [-1, 1]
    w_q *= 2
    w_q -= 1

    return w_q


def LogQuant_A(x, numBits, bounds):
    if bounds == 'two_sided':
        x = torch.clamp(x, min=-1.0, max=1.0)
    elif bounds == 'one_sided':
        x = torch.clamp(x, min=0, max=1.0)

    if numBits == 32:
        return x

    w_q = LogQuant.apply(x, numBits, bounds)

    return w_q


class LogQuant(Function):
    @staticmethod
    def forward(ctx, x, numBits, bounds):
        s = torch.sign(x)
        s[s == 0] = 1

        min_vals = {2: 0.33, 3: 0.15, 4: 0.07, 5: 0.05, 6: 0.03}
        min = min_vals[numBits] if numBits <= 6 else 0.02

        w_q = torch.clamp(torch.abs(x), min=min)        # Pick a min not to close to zero
        w_q = torch.log2(w_q)
        w_q = min_max_quantize(w_q, numBits, bounds)
        w_q = torch.pow(2, w_q) * s

        return w_q

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def min_max_quantize(input, bits, bounds):
    min_val, max_val = input.min(), input.max()

    # Rescale to [0,1]
    input_rescale = (input - min_val) / (max_val - min_val)

    if bounds == 'two_sided':
        n = math.pow(2.0, bits-1) - 1       # One bit will be consumed by the sign
    elif bounds == 'one_sided':
        n = math.pow(2.0, bits) - 1

    input_rescale = input_rescale * n
    v = torch.round(input_rescale) / n

    v = v * (max_val - min_val) + min_val
    return v


class RoundNoGradient(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g

