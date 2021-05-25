import torch
import numpy as np

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

    rate = torch.logdet(I.unsqueeze(0) + a * W.matmul(
        W.transpose(2, 1))).sum() / (n * c)

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

        rate = rate + torch.logdet(I.unsqueeze(0) + a * W_k.matmul(
            torch.diag(Pi[i])).matmul(W_k.transpose(2, 1))).sum() / (m*n*c) * (trPi)

    rate = rate / 2

    return rate

def mcr2_loss(input, target, beta=1.0, device='cuda', eps=0.1):
    n_class = target.max() + 1

    m = input.shape[0]
    Pi = np.zeros(shape=(n_class, m))
    for j in range(len(target)):
        k = target[j]
        Pi[k, j] = 1.
    Pi = torch.tensor(Pi, dtype=torch.float64).to(device)
    input = input.double()

    return - MultiChannelRateDistortion(input, device, eps) + \
           beta * MultiChannelRateDistortion_Label(input, Pi, device, eps)

if __name__ == '__main__':
    inp = torch.randn(100,10, 5, 5).to('cuda:0')
    inp.requires_grad_(True)
    loss = mcr2_loss(
        input=inp,
        target=torch.randint(low=0, high=10, size=(100,))
    )
    print(inp)
    loss.backward()
    print(inp.grad)
    print(loss)
