import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class KL_Divergence(nn.Module):
# class KL_Divergence(Function):
    def __init__(self):
        super(KL_Divergence, self).__init__()

    def forward(self, p2, p1):
        # Add epsilon to avoid division over zero
        epsilon = 0.00001

        # All operations are element wise
        D_KL = (1.0/(p2+epsilon).shape[0])*torch.sum((p2+epsilon)*torch.log((p2+epsilon)/(p1+epsilon)))

        return D_KL

