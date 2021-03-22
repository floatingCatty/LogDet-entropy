import torch
import numpy as np

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

    W = W.reshape(m, c, -1).permute(1,2,0)
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
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(0)
    W = W.transpose(1, 0)
    n = W.shape[0]
    a = n / (m * eps ** 2)

    rate = torch.logdet(torch.eye(n).to(device) + a * torch.matmul(W, W.T))

    return rate / 2

def entropyRD(W, device='cpu', eps=0.1):
    m = W.shape[0]

    W = W.reshape(m, -1)
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(0)
    W = W.transpose(1, 0)
    n = W.shape[0]
    a = n / (m * eps ** 2)

    rate = torch.logdet(torch.eye(n).to(device) + a * torch.matmul(W, W.T))
    rate = rate - n * torch.log(torch.scalar_tensor(n / 6.28 * eps ** 2)) + n

    return rate / 2

def MCentropyRD(W, device='cpu', eps=0.1):
    # W in R(M, C, D)
    m = W.shape[0]
    c = W.shape[1]
    d = W.shape[2]

    W = W.reshape(m, -1)
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(0)
    W = W.transpose(1, 0)
    n = W.shape[0]
    a = n / (m * eps ** 2)

    rate = torch.logdet(torch.eye(n).to(device) + a * torch.matmul(W, W.T))
    rate = rate - d/m * torch.log(torch.scalar_tensor(c))
    rate = rate - d * torch.log(torch.scalar_tensor(d / 6.28 * eps ** 2)) + d

    return rate / 2

def MI(X,Y, device='cpu', eps=0.1):
    return RateDistortion(X,device,eps)+RateDistortion(Y,device,eps)-MutualSpace(X,Y,device,eps)

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
        mean = W_k.sum(dim=1) / trPi
        W_k = W - mean.unsqueeze(1)
        # rate = rate + torch.logdet(torch.eye(n).to(device) + a * W_k.matmul(
        #     torch.diag(Pi[i])).matmul(W_k.T)) / (m*n) * (trPi + n)
        # rate = rate + torch.log(1 + mean.dot(mean) / eps ** 2) / m

        rate = rate + torch.logdet(torch.eye(n).to(device) + a * W_k.matmul(
            torch.diag(Pi[i])).matmul(W_k.T)) / m * trPi

    rate = rate / 2.

    return rate

def RD_fn(X, W, Label, device='cpu', eps=0.1):
    n_class = Label.max() + 1
    rate = []
    m = W.shape[0]
    Pi = np.zeros(shape=(n_class, m))
    for j in range(len(Label)):
        k = Label[j]
        Pi[k, j] = 1.

    Pi = torch.tensor(Pi, dtype=torch.float64).to(device)
    W = W.double().detach()
    X = X.double().detach()

    rate.append(RateDistortion(W, device, eps))
    rate.append(RateDistortion_Label(W, Pi, device, eps))
    rate.append(MultiChannelRateDistortion(W, device, eps))

    rate.append(MutualSpace(W=W, V=X, device=device, eps=eps))

    return torch.tensor(rate)



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

def mr():
    '''
    record the measurement idea
    1. how the inforamtion about label help saved space than those with random labels?
    2. how the neural network training influence about the feature space?
    3. what can conv layers get from random noise?
    '''

    pass

def MutualSpace(W, V, device='cpu', eps=0.1):
    '''
    This is the alpha version of matrix MR
    W should be in shape R(M, C1, H, W)
    V should be in shape R(M, C2, h, w)
    '''

    m = W.shape[0]

    W = W.reshape(m, -1)
    V = V.reshape(m, -1)
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(dim=0)
    mean = V.mean(dim=0)
    V = V - mean.unsqueeze(dim=0)

    # W in shape(c, m, n)
    I = torch.eye(m).to(device)
    a1 = np.sqrt(W.shape[1] / (m * eps ** 2))
    a2 = np.sqrt(V.shape[1] / (m * eps ** 2))

    W = torch.cat((a1*W, a2*V), dim=1)

    rate = torch.logdet(
        I + W.matmul(W.T)
    )

    torch.cuda.empty_cache()

    return rate / 2

# def MutualSpace_(W, V, device='cpu', eps=0.1):
#     '''
#     This is the alpha version of matrix MR
#     W should be in shape R(M, C1, H, W)
#     V should be in shape R(M, C2, h, w)
#     '''
#
#     m = W.shape[0]
#
#     W = W.reshape(m, -1)
#     V = V.reshape(m, -1)
#     mean = W.mean(dim=0)
#     W = W - mean.unsqueeze(dim=0)
#     mean = V.mean(dim=0)
#     V = V - mean.unsqueeze(dim=0)
#
#     # W in shape(c, m, n)
#     I = torch.eye(m).to(device)
#     a1 = W.shape[1] / (m * eps ** 2)
#     a2 = V.shape[1] / (m * eps ** 2)
#
#     rate = torch.logdet(
#         I + a1*W.matmul(W.T) + a2*V.matmul(V.T)
#     )
#
#     torch.cuda.empty_cache()
#
#     return rate / 2


def RateDistortion_(W, device='cpu', eps=0.1):
    m = W.shape[0]
    W = W.reshape(m, -1)
    mean = W.mean(dim=0)
    W = W - mean.unsqueeze(0)

    W = W.transpose(1, 0)

    n = W.shape[0]

    a = n / (m * eps ** 2)
    # rate = torch.logdet(torch.eye(m).to(device) + a * torch.matmul(W.T, W)) / (m*n) * (m+n)
    # rate = rate + torch.log(1+mean.dot(mean)/eps**2) / m
    rate = torch.logdet(torch.eye(n).to(device) + a * torch.matmul(W, W.T))
    rate = rate / 2.

    return rate


if __name__ == '__main__':
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    rate = []
    print(MutualSpace(W=torch.randn(100,20), V=torch.ones(100,20)+torch.randn(100,20)*0.1, device='cpu'))









    # print(RateDistortion(a+b, device='cuda:0'))
    # print(RateDistortion(a+a, device='cuda:0'))


