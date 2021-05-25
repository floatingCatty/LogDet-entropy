import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RD_fn, mi

class LeNet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2)

        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x, sample=None, label=None, device='cpu', return_rate=False):

        rate = []
        temp = []
        Channel = []
        temp.append(sample)
        if return_rate:
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = F.relu(self.fc2(x))
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = self.fc3(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))

            for i in range(len(temp)-1):
                Channel.append(mi(temp[i], temp[i+1],device,'cov'))

            return x, torch.stack(rate), torch.stack(Channel)
        else:
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features