import torch
import torch.nn as nn


def cmm(A, B):
    return A.conj().mm(B)

class ChopfieldAttention(nn.Module):
    def __init__(self, beta, d_Q, d_K, d_V):
        super(ChopfieldAttention, self).__init__()
        self.beta = beta
        self.W_Q = nn.Parameter(torch.randn(size=(d_Q, d_Q), dtype=torch.complex64))
        self.W_K = nn.Parameter(torch.randn(size=(d_K, d_K), dtype=torch.complex64))
        self.W_V = nn.Parameter(torch.randn(size=(d_V, d_V), dtype=torch.complex64))

    def forward(self, R, Y):
        Y = Y.mm(self.W_K)
        Z = self.beta * cmm(R.mm(self.W_Q), Y.T)
        Z = torch.softmax()
        # softmax 这里需要把 x 和 x*区分开，才能实现有向图查询（非对称查询）



# a = torch.rand((3,2), dtype=torch.complex64)
# print(a.conj().mm(a.T))