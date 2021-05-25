import torch
import torch.nn.functional as F
import numpy as np

def RD_fn(X, T, Label, device='cuda'):
    n_class = Label.max() + 1
    rate = []
    m = T.shape[0]
    Pi = np.zeros(shape=(n_class, m))
    for j in range(len(Label)):
        k = Label[j]
        Pi[k, j] = 1.
    X = X.double().detach()
    T = T.double().detach()
    X = X.reshape(m,-1)
    T = T.reshape(m,-1)
    Pi = torch.tensor(Pi, dtype=torch.float64).to(device)
    beta = 100*X.shape[1]

    rate.append(LogDet(X, device, 'pearson', beta=beta))
    rate.append(LogDet(T, device, 'pearson', beta=beta))
    rate.append(LogDet(Pi.T, device, 'pearson', beta=beta))
    rate.append(join(device, 'pearson',beta, X, T))
    rate.append(join(device, 'pearson',beta, T, Pi.T))

    rate.append(LogDet(X, device, 'cov', beta=beta))
    rate.append(LogDet(T, device, 'cov', beta=beta))
    rate.append(LogDet(Pi.T, device, 'cov', beta=beta))
    rate.append(join(device, 'cov',beta, X, T))
    rate.append(join(device, 'cov',beta, T, Pi.T))

    return torch.stack(rate)

def cal_pearson(A, device='cuda', beta=1.):
    d = A.shape[1]

    A = A.T.matmul(A) / (A.norm(dim=0).unsqueeze(1).matmul((A.norm(dim=0)).unsqueeze(0)) + 1e-8)
    # A = A.masked_fill(torch.abs(A).lt(0.5), 0)
    A = torch.eye(d).to(device) + beta*A

    return A

def cal_distance(A, device='cuda', beta=1.0):
    A = A.T.matmul(A) / (A.norm(dim=0).unsqueeze(1).matmul((A.norm(dim=0)).unsqueeze(0)) + 1e-8)
    # A = torch.cosine_similarity(A,A,dim=0)
    # x = x - x.mean(dim=0).unsqueeze(0)
    # A = A.T.matmul(A)
    A = torch.exp(A-1)

    return beta * A + torch.eye(A.shape[0]).to(device)

def cal_cov(A, device='cuda', beta=1.):
    d = A.shape[1]
    n = A.shape[0]

    A = A.T.matmul(A) / n
    # A = A.masked_fill(torch.abs(A).lt(0.5), 0)
    A = torch.eye(d).to(device) + beta*A

    return A

def LogDet(A, device='cuda', mode='pearson', beta=1.):
    A = A.reshape(A.shape[0], -1)
    # A in R(n,d)
    A = A - A.mean(dim=0).unsqueeze(0)
    if mode == 'pearson':
        A = cal_pearson(A, device, beta)
    elif mode == 'cov':
        A = cal_cov(A, device, beta)
    elif mode == 'pos':
        A = cal_distance(A, device, beta)

    result = torch.logdet(A)
    return result / 2

def mi(A,B,device='cuda',mode='pearson', beta=False):
    A = A.reshape(A.shape[0],-1)
    B = B.reshape(B.shape[0],-1)
    if not beta:
        beta = 100*(A.shape[1]+B.shape[1])

    return LogDet(A,device, mode,beta) + LogDet(B, device, mode,beta) - join(device,mode, beta,A,B)

def join(device,mode, beta=1.,*kwargs):
    Z = kwargs[0]
    m = Z.shape[0]
    Z = Z.reshape(m,-1)
    for i in kwargs[1:]:
        Z = torch.cat((Z,i.reshape(m,-1)), dim=1)

    return LogDet(Z, device, mode,beta)


def mi_loss(input, org, device, beta=10):
    input = input.reshape(input.shape[0], -1)
    org = org.reshape(org.shape[0], -1)
    beta = 100 * (input.shape[1] + input.shape[1])

    return join(device, 'cov', beta, input, org) - LogDet(A=input, device=device, mode='cov', beta=beta)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # o = torch.ones(512, 10).to('cuda')
    # b = torch.randint(low=0, high=10, size=(512,))
    # Pi = torch.ones(10, 512)
    # for j in range(len(b)):
    #     k = b[j]
    #     Pi[k, j] = 0.
    # Pi = Pi.T.to('cuda')
    # print(mi_loss(input=Pi, target=b, device='cuda', beta=1000))
    # print(mi_loss(input=o, target=b, device='cuda', beta=1000))
    # print(mi_loss(input=o, target=b, device='cuda', beta=100))

    # c = torch.randn(1000,50).to('cuda')
    # print(join(a,b,b))
    # print(join(a,b,c))
    # b = torch.randn(100,10)
    # e = []
    # for i in tqdm(range(1,1000,5)):
    #     a = torch.randn(i, 100)
    #     e.append(LogDet(a))
    #
    # plt.plot(e)
    # plt.show()

    a = torch.randn(10,1,28,28)
    b = torch.randn(10,784)

    print(join('cpu', 'pearson',100*784,a,b))