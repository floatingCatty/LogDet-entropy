import torch
import torch.nn as nn
from utils import RD_fn

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 'M'], [512, 512, 'M']],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x, sample=None, label=None, device='cpu', return_rate=False):
        rate = []

        if return_rate:
            for layer in self.features:
                x = layer(x)
                rate.append(RD_fn(X=sample, W=x, Label=label, device=device))
        else:
            for layer in self.features:
                x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if return_rate:
            return x, torch.stack(rate)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layer(params, batch_norm=True):
    layers = []

    in_channel = 3

    for param in params:
        layers.append(layer(params=param, in_channels=in_channel, batch_norm=batch_norm))
        in_channel = param[1]

    return nn.ModuleList(layers)

class layer(nn.Module):
    def __init__(self, params, in_channels=3, batch_norm=True):
        super(layer, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, params[0], kernel_size=3, padding=1)
        self.cnn2 = nn.Conv2d(params[0], params[1], kernel_size=3, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        if batch_norm:
            self.norm = nn.BatchNorm2d(params[1])
        self.relu = nn.ReLU(inplace=True)

        self.batch_norm = batch_norm

    def forward(self, x):
        x = self.cnn2(self.cnn1(x))

        if self.batch_norm:
            x = self.norm(x)
        
        x = self.relu(x)

        return x

def vgg13(num_classes, batch_norm=True):
    feature = make_layer(cfgs['B'], batch_norm=batch_norm)

    return VGG(feature, num_classes=num_classes)


if __name__ == '__main__':
    model = vgg13(num_classes=10)
    print(model.parameters())