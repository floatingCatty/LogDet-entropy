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

    if mode == 'MNIST':
        trainset = torchvision.datasets.MNIST(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.MNIST(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode == 'FashionMNIST':
        trainset = torchvision.datasets.FashionMNIST(
            root=PATH, train=True,
            download=True, transform=transform
        )
        testset = torchvision.datasets.FashionMNIST(
            root=PATH, train=False,
            download=True, transform=transform
        )

    elif mode == 'ImageNet':
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

def rateReduction(V, e=0.1):
    N = V.shape[0]
    V = V.reshape(N, -1)
    V = cp.array(V.detach().cpu().numpy())
    V = V.transpose(1,0)

    m = V.shape[1]
    C = V.shape[0]

    a = C / (m * e**2)
    r2 = cp.linalg.slogdet(cp.eye(C) + a * cp.matmul(V, V.T))[1] / 2.
    return r2

def rateReductionWithLabel(V, label, e=0.1):
    n_class = label[label.argmax()] + 1
    N = V.shape[0]
    Pi = np.zeros(shape=(n_class, N, N))
    for j in range(len(label)):
        Pi[label[j], j, j] = 1.
    Pi = cp.array(Pi)


    V = V.reshape(N, -1)
    V = cp.array(V.detach().cpu().numpy())
    V = V.transpose(1, 0)

    C = V.shape[0]
    compress_loss = 0.

    for j in range(n_class):
        trPi = cp.trace(Pi[j]) + 1e-8
        a = C / (trPi * e**2)
        r2 = cp.linalg.slogdet(cp.eye(C) + a * cp.matmul(cp.matmul(V, Pi[j]), V.T))[1]
        compress_loss += r2 * trPi / N

    return  compress_loss / 2

def MCR2_loss(V, label, e=0.1):
    '''
    V is type of torch tensor, in R(m, c)
    label is a list of m items
    '''
    outter = rateReduction(V, e)
    inner = rateReductionWithLabel(V, label, e)

    return [outter, inner]




if __name__ == '__main__':
    V1 = torch.randn(10, 128*8*8)
    print(rateReduction(V1, 0.1))
    # print(rateReductionWithLabel(V1, label, 0.1))

