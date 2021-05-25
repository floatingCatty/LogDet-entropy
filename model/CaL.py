import torch.nn as nn
import torch.nn.functional as F
from utils import RD_fn, mi

class basicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act='ReLU'):
        super(basicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        if act == 'ReLU':
            self.act = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = F.layer_norm(x, normalized_shape=x.shape[1:])
        x = self.act(x)
        x = self.maxpool(x)


        return x

class CaLnet(nn.Module):
    def __init__(self, in_channel, num_classes, n_Layer, n_Channel, act):
        super(CaLnet, self).__init__()
        self.layer = []
        n_Channel = n_Channel.split(',')
        for i in range(n_Layer):
            self.layer.append(basicBlock(in_channel=in_channel, out_channel=int(n_Channel[i]), act=act))
            in_channel = int(n_Channel[i])
        self.layer = nn.ModuleList(self.layer)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Linear(int(n_Channel[-1])*36, num_classes)

    def forward(self, x, sample=None, label=None, device='cpu', return_rate=False):
        rate = []
        temp = []
        Channel = []
        bsz = x.shape[0]
        temp.append(sample)

        if return_rate:
            for layer in self.layer:
                x = layer(x)
                temp.append(x.detach())
                rate.append(RD_fn(T=x, X=sample, Label=label, device=device))

            x = self.avgpool(x)
            x = self.classifier(x.view(bsz, -1))

            for i in range(len(temp)-1):
                Channel.append(mi(temp[i], temp[i+1],device,'cov'))

            return x, torch.stack(rate), torch.stack(Channel)

        else:
            for layer in self.layer:
                x = layer(x)


        x = self.classifier(self.avgpool(x).view(bsz, -1))

        return x

if __name__ == '__main__':
    import torch
    a = torch.randn(1, 3, 5, 5)
    mp = nn.MaxPool2d(kernel_size=3, stride=2)
    print(mp(a).size())