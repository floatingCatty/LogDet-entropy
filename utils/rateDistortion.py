import torch
import numpy as np
from utils.LogDet import *

'''
tips in mind:
1. should we split each channel to compute? it depends on the result
2. is the cp result the same with torch? yes √
3. specific additions should be concerned if input is not zero-mean.
4. should we multiply R(D) with (m+n) then divided by (m*n) to keep 
    the maximum rate for each variable holds, to compare the rate increament
    across layers?
5. up flow Q -_-
'''


def MultiChannelRateDistortion(W, device='cpu', eps=0.1):
    '''
    W in shape(N, C, H, W)
    '''
    m, c, _, _ = W.shape

    W = W.reshape(m, c, -1).permute(1, 2, 0)
    mean = W.mean(dim=2)
    W = W - mean.unsqueeze(dim=2)

    # W in shape(c, n, m)
    n = W.shape[1]
    I = torch.eye(n).to(device)
    a = n / (m * eps ** 2)
    # rate = torch.logdet(I.unsqueeze(0) + a * W.matmul(W.transpose(2,1))).sum() / (m*n*c) * (m+n)
    # rate = rate + torch.log(1 + mean.mul(mean).sum(dim=1)/eps**2).sum() / (m*c)
    rate = torch.logdet(I.unsqueeze(0) + a * W.matmul(
        W.transpose(2, 1))).mean()

    rate = rate / 2.

    return rate


def MultiChannelRateDistortion_Label(W, Pi, device='cpu', eps=0.1):
    m, c, _, _ = W.shape

    W = W.reshape(m, c, -1).permute(1, 2, 0)

    k, _ = Pi.shape
    n = W.shape[1]
    I = torch.eye(n).to(device)
    # W in shape(c, n, m)
    rate = 0
    for i in range(k):
        trPi = Pi[i].sum() + 1e-8
        a = n / (trPi * eps ** 2)
        W_k = W.matmul(torch.diag(Pi[i]))
        mean = W_k.sum(dim=2) / trPi
        W_k = W_k - mean.unsqueeze(2)

        # rate = rate + (torch.logdet(I.unsqueeze(0) + a * W_k.matmul(torch.diag(Pi[i])).matmul(W_k.transpose(2,1))) / (m*n*c)).sum() * (trPi+n)
        # rate = rate + torch.log(1 + mean.mul(mean).sum(dim=1)/eps**2).sum() / (m*c)

        rate = rate + torch.logdet(I.unsqueeze(0) + a * W_k.matmul(
            torch.diag(Pi[i])).matmul(W_k.transpose(2, 1))).mean() / m * (trPi)

    rate = rate / 2

    return rate


def RateDistortion(W, device='cpu', eps=0.1):
    m = W.shape[0]

    W = W.reshape(m, -1)
    mean = W.mean(dim=1)
    W = W - mean.unsqueeze(1)
    # W = W / (torch.sqrt((W**2).sum(dim=1)).unsqueeze(1) + 1e-8)
    n = W.shape[1]
    a = n / (m * eps ** 2)

    rate = torch.logdet(torch.eye(m).to(device) + a * torch.matmul(W, W.T))

    return rate / 2


def RateDistortion_Label(W, Pi, device='cpu', eps=0.1):
    m = W.shape[0]
    W = W.reshape(m, -1)

    k, _ = Pi.shape

    W = W.transpose(1, 0)

    n = W.shape[0]

    rate = 0
    for i in range(k):
        trPi = Pi[i].sum() + 1e-8
        a = n / (trPi * eps ** 2)
        W_k = W.matmul(torch.diag(Pi[i]))
        mean = W_k.sum(dim=0) / trPi
        W_k = W_k - mean.unsqueeze(0)
        # W_k = W_k / (torch.sqrt((W_k ** 2).sum(dim=0)).unsqueeze(0) + 1e-8)

        rate = rate + torch.logdet(torch.eye(n).to(device) + a * W_k.matmul(W_k.T)) / m * trPi

    rate = rate / 2.

    return rate




def MCR2_loss(input, target, beta=1.0, device='cuda', eps=0.1):
    n_class = target.max() + 1

    m = input.shape[0]
    Pi = np.zeros(shape=(n_class, m))
    for j in range(len(target)):
        k = target[j]
        Pi[k, j] = 1.
    Pi = torch.tensor(Pi, dtype=torch.float64).to(device)
    input = input.double()

    return MultiChannelRateDistortion(input, device, eps) - \
           beta * MultiChannelRateDistortion_Label(input, Pi, device, eps)


def JointEntropy(X, Y, device='cpu', eps=0.1):
    '''
    This is the alpha version of matrix MR
    W should be in shape R(M, C1, H, W)
    V should be in shape R(M, C2, h, w)
    '''
    m = X.shape[0]

    X = X.reshape(m, -1)
    Y = Y.reshape(m, -1)

    X = torch.cat((X, Y), dim=1)
    mean = X.mean(dim=0)
    X = X - mean.unsqueeze(dim=0)

    # W in shape(c, m, n)
    I = torch.eye(m).to(device)
    a = X.shape[1] / (m * eps ** 2)

    rate = torch.logdet(
        I + a * X.matmul(X.T)
    )

    torch.cuda.empty_cache()

    return rate / 2


def MutualInformation(X, Y, device='cpu', eps=0.1):
    m = X.shape[0]
    X = X.reshape(m, -1)
    Y = Y.reshape(m, -1)

    mean = X.mean(dim=0)
    X = X - mean.unsqueeze(0)
    # X = X / (torch.sqrt((X**2).sum(dim=0)).unsqueeze(0)+1e-8)
    mean = Y.mean(dim=0)
    Y = Y - mean.unsqueeze(0)
    # Y = Y / (torch.sqrt((Y**2).sum(dim=0)).unsqueeze(0)+1e-8)

    I = torch.eye(m).to(device)
    a = (X.shape[1] + Y.shape[1]) / (m * eps ** 2)
    mi = 0.5 * torch.logdet(I + a * torch.matmul(X, X.T))
    mi = mi + 0.5 * torch.logdet(I + a * torch.matmul(Y, Y.T))
    mi = mi - JointEntropy(X, Y, device, eps)

    torch.cuda.empty_cache()

    return mi


if __name__ == '__main__':
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import time

    a = torch.randn(1000, 1000).to('cuda')
    b = torch.randn(1000, 1000).to('cuda')
    print(JointEntropy(X=a, Y=b, device='cuda'))
    print(MutualInformation(X=a, Y=b, device='cuda'))
    # M = []
    # j = 0
    # for i in tqdm(range(50)):
    #     s = (1 - 2*i/100) * a + 2*i/100*b
    #     M.append(MI(X=s, Y=b))
    #     j += 20
    # plt.plot(M)
    #
    # plt.show()

    # print(RateDistortion(a, device='cpu', eps=0.2))
    # print(RateDistortion(a+a, device='cuda:0'))


