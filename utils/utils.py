import torch
import numpy as np
import cupy as cp
import torchvision
import sys
from torchvision import transforms

def generateNormalizedClusteredData(cluster, std, num):
    N = len(cluster)
    data = torch.normal(mean=torch.zeros(num, N), std=std) + cluster

    norm = data.mul(data).sum(dim=1).sqrt().view(num, 1).expand(num, N)

    return  data / norm

def top_n(pre, label, n):
    # pre in np type of R(m,k)
    topn = pre.argsort(axis=1)[:,-1:-1-n:-1]

    acc = np.mean(np.array([1 if label[i] in topn[i] else 0 for i in range(len(label))]))

    return acc

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def relu(inX):
    return cp.maximum(0, inX)

def numpy_conv(inputs,filter,_result,padding="VALID"):
    H, W = inputs.shape
    filter_size = filter.shape[0]
    # default np.floor
    filter_center = int(filter_size / 2.0)
    filter_center_ceil = int(np.ceil(filter_size / 2.0))

    result = np.zeros((_result.shape))

    H, W = inputs.shape
    for r in range(0, H - filter_size + 1):
        for c in range(0, W - filter_size + 1):
            # 池化大小的输入区域
            cur_input = inputs[r:r + filter_size,
                        c:c + filter_size]
            cur_output = cur_input * filter
            conv_sum = np.sum(cur_output)
            result[r, c] = conv_sum
    return result

def useData(mode, transform, PATH='/home/17320015070/notespace/dataset'):

    if mode is 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode is 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode is 'FashionMNIST':
        trainset = torchvision.datasets.FashionMNIST(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode is 'ImageNet':
        trainset = torchvision.datasets.ImageNet(
            root=PATH, train=True,
            download=True, transform=transform
        )

        testset = torchvision.datasets.ImageNet(
            root=PATH, train=False,
            download=True, transform=transform
        )

    else:
        raise ValueError

    return trainset, testset

def getConfig():
    return sys.argv[1]