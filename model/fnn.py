import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import RD_fn

class FNN(nn.Module):
    def __init__(self, act='ReLU'):
        super(FNN,self).__init__()
        self.l1 = nn.Linear(784,1024)
        self.l2 = nn.Linear(1024, 20)
        self.l3 = nn.Linear(20, 20)
        self.l4 = nn.Linear(20, 20)
        self.l5 = nn.Linear(20, 10)
        if act == 'ReLU':
            self.gate = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.gate = nn.Sigmoid()
        elif act == 'tanh':
            self.gate = nn.Tanh()
        elif act == 'gelu':
            self.gate = nn.GELU()
        elif act == 'lrelu':
            self.gate = nn.LeakyReLU()
        else:
            raise ValueError

    def forward(self, x, sample=None, label=None, device='cpu', return_rate=False):
        rate = []
        x = x.view(-1,784) # Flattern the (n,3,32,32) to (n,3096)

        if return_rate:
            x = self.gate(self.l1(x))
            rate.append(RD_fn(X=sample, W=x, Label=label, device=device))
            x = self.gate(self.l2(x))
            rate.append(RD_fn(X=sample, W=x, Label=label, device=device))
            x = self.gate(self.l3(x))
            rate.append(RD_fn(X=sample, W=x, Label=label, device=device))
            x = self.gate(self.l4(x))
            rate.append(RD_fn(X=sample, W=x, Label=label, device=device))

            return self.l5(x), torch.stack(rate)
        else:
            x = self.gate(self.l1(x))
            x = self.gate(self.l2(x))
            x = self.gate(self.l3(x))
            x = self.gate(self.l4(x))

        return self.l5(x)