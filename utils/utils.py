import torch
import numpy as np
import torchvision
import sys
from torchvision import transforms

def generateNormalizedClusteredData(cluster, std, num):
    N = cluster.shape[1]
    out = []
    for i in range(cluster.shape[0]):
        data = torch.normal(mean=torch.zeros(num, N), std=std) + cluster[i]
        norm = data.mul(data).sum(dim=1).sqrt().view(num, 1).expand(num, N)
        data = data / norm
        out.append(data)

    return torch.stack(out, dim=0).view(-1, N)

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
    elif mode == 'STL':
        trainset = torchvision.datasets.STL10(
            root=PATH, split='train',
            download=True, transform=transform
        )

        testset = torchvision.datasets.STL10(
            root=PATH, split='test',
            download=True, transform=transform
        )

    else:
        raise ValueError

    return trainset, testset

def getConfig():
    return sys.argv[1]


def loss_fn(input, target, n_class=10, device='cuda'):
    # input in R(bsz, d)
    # target in R(bsz)
    m = input.shape[0]
    Pi = np.zeros(shape=(n_class, m, m), dtype=np.float32)
    for j in range(len(target)):
        k = target[j]
        Pi[k, j, j] = 1.

    Pi = torch.tensor(Pi).to(device)
    loss = Pi.matmul(input)
    loss = torch.einsum('nmd,pqd->npmq',loss, loss).sum(dim=(2,3))
    loss = loss*(1-torch.eye(n_class)).to(device)

    return torch.abs(loss.sum()) / m

def combine(oldPATH, newPATH):
    fold = torch.load(oldPATH, map_location='cpu')
    fnew = torch.load(newPATH, map_location='cpu')

    # fold['data'][:, :, 2, :] = fold['data'][:,:,0,:]
    fold['data'][:,:,2,:] = fnew['data'][:,:,0,:]

    torch.save(obj=fold, f=oldPATH)

    return True

if __name__ == '__main__':
    a = torch.randn(100,150).to('cuda')
    t = torch.randint(0,10,(100,))
    combine(
        oldPATH="E:\\thesis\MCR2\log\\analysisData\degenerativeCIFAR10.pth",
        newPATH="E:\\thesis\MCR2\log\\analysisData\degenerative_LogDetCIFAR10.pth"
    )







# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     # rate = []
#     # V1 = torch.randn(100, 10, 10, 10) * 0.3 + torch.randn(1, 10, 10, 10)
#     # V2 = torch.randn(100, 10, 10, 10) * 0.3 + torch.randn(1, 10, 10, 10)
#     #
#     # r = 0
#     # for i in range(10):
#     #     r += rateReduction(V1[:, i, :, :].unsqueeze(1), 0.1)
#     # rate.append(r)
#     # for i in range(100):
#     #     V1[i] = V2[i]
#     #     r = 0
#     #     for j in range(10):
#     #         r += rateReduction(V1[:,j,:,:].unsqueeze(1), 0.1)
#     #     rate.append(r)
#     #
#     # plt.plot(rate)
#     #
#     # plt.show()
#     cp.cuda.Device(0).use()
#     r = []
#     for m in [10000, 20000, 30000]:
#         for n in range(1,1000):
#             r.append(rateReduction(torch.randn(m, n), e=0.01) * (n+m) / (m*n))
#
#         plt.plot(r)
#         r = []
#     plt.savefig('IMG.png', dpi=600)
#     plt.show()


