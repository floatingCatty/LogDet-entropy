from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# On the Information Plane of Autoencoders

def pairwise_distances(x):
    # x should be two dimensional
    x = x.view(x.shape[0], -1)
    # x = x - x.mean(dim=0).unsqueeze(0)
    x = x / (x.norm(dim=0).unsqueeze(0) + 1e-8)
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_gram_mat(x, sigma):
    dist = pairwise_distances(x)
    # dist = dist/torch.max(dist)
    return torch.exp(-dist / sigma)


def reyi_entropy(x, sigma):
    alpha = 1.01
    k = calculate_gram_mat(x, sigma)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))
    return entropy


def joint_entropy(x, y, s_x, s_y):
    alpha = 1.01
    x = calculate_gram_mat(x, s_x)
    y = calculate_gram_mat(y, s_y)
    k = torch.mul(x, y)
    k = k / torch.trace(k)
    eigv = torch.abs(torch.symeig(k, eigenvectors=True)[0])
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow))

    return entropy


def calculate_MI(x, y, s_x, s_y):
    Hx = reyi_entropy(x, sigma=s_x)
    Hy = reyi_entropy(y, sigma=s_y)
    Hxy = joint_entropy(x, y, s_x, s_y)
    Ixy = Hx + Hy - Hxy
    # normlize = Ixy/(torch.max(Hx,Hy))

    return Ixy


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x, s_y):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


def loss_fn(inputs, outputs, targets, batch_size, name):
    inputs_2d = inputs.view(batch_size, -1)
    error = F.softmax(outputs, dim=1) - F.one_hot(targets, 10)
    if name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
    if name == 'mse':
        criterion = nn.MSELoss()
        loss = criterion(F.softmax(outputs, dim=1), F.one_hot(targets, 10))
    if name == 'HSIC':
        loss = HSIC(inputs_2d, error, s_x=1, s_y=1)
    if name == 'DIB':
        loss = calculate_MI(inputs_2d, error, s_x=1, s_y=1)
    if name == 'MEE':
        loss = reyi_entropy(error, sigma=1)
    return loss

def RateDistortion(W, device='cpu', eps=0.1):
    m = W.shape[0]

    W = W.reshape(m, -1)
    n = W.shape[1]
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(0)
    sigma = 2 * np.sqrt(W.shape[1]) * W.shape[0] ** (-1 / (4 + W.shape[1]))
    K = calculate_gram_mat(W, sigma)
    a = n / (m * eps ** 2)

    rate = torch.logdet(torch.eye(m).to(device) + a * K)
    rate = rate / 2.

    return rate


if __name__ == '__main__':
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        data = []
        org = torch.ones(2000, 200)
        noise = torch.randn_like(org)
        for i in range(0,100,5):
            a = noise*(i/100) + org*(1-i/100)
            data.append(RateDistortion(a, 'cpu', 0.1))

        plt.plot(data)
        plt.show()


