import torch.nn as nn
import torch.nn.functional as F
from utils import MCR2_loss

class FNN(nn.Module):
    def __init__(self, gate='relu'):
        super(FNN,self).__init__()
        self.l1 = nn.Linear(3072,1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)
        if gate == 'relu':
            self.gate = nn.ReLU()
        elif gate == 'sigmoid':
            self.gate = F.sigmoid
        elif gate == 'tanh':
            self.gate = nn.Tanh()
        elif gate == 'gelu':
            self.gate = nn.GELU()
        elif gate == 'lrelu':
            self.gate = nn.LeakyReLU()
        else:
            raise ValueError

    def forward(self, x, y=None, return_rate=False):
        rate = []
        x = x.view(-1,3072) # Flattern the (n,3,32,32) to (n,3096)

        if return_rate:
            x = self.gate(self.l1(x))
            rate.append(MCR2_loss(V=x, label=y))
            x = self.gate(self.l2(x))
            rate.append(MCR2_loss(V=x, label=y))
            x = self.gate(self.l3(x))
            rate.append(MCR2_loss(V=x, label=y))
            x = self.gate(self.l4(x))
            rate.append(MCR2_loss(V=x, label=y))

            return self.l5(x), rate
        else:
            x = self.gate(self.l1(x))
            x = self.gate(self.l2(x))
            x = self.gate(self.l3(x))
            x = self.gate(self.l4(x))

        return self.l5(x)