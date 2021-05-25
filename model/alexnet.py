import torch
import torch.nn as nn
from utils import RD_fn, mi

class AlexNet(nn.Module):

    def __init__(self, act='ReLU', num_classes=1000):
        super(AlexNet, self).__init__()
        if act == 'ReLU':
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(192),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2),
            )
        elif act == 'tanh':
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.Tanh(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.Tanh(),
                nn.BatchNorm2d(192),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.BatchNorm2d(384),
                nn.Tanh(),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.Tanh(),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.Tanh(),
                # nn.MaxPool2d(kernel_size=3, stride=2),
            )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x, sample=None, label=None, device='cpu', return_rate=False):
        rate = []
        temp = []
        Channel = []
        temp.append(sample)
        if return_rate:
            x = self.layer1(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = self.layer2(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = self.layer3(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = self.layer4(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))
            x = self.layer5(x)
            temp.append(x.detach())
            rate.append(RD_fn(X=sample, T=x, Label=label, device=device))

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)

            for i in range(len(temp)-1):
                Channel.append(mi(temp[i], temp[i+1],device,'cov'))

            return x, torch.stack(rate), torch.stack(Channel)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)


        return x