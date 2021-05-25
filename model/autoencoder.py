import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn

class autoencoder:
    def __init__(self, eps):
        self.eps = eps**2

    def XX(self, X):
        return X.matmul(X.T)

    def gt(self, TT, XX, T, alpha):
        return alpha*(torch.inverse(self.eye+alpha*TT) - torch.inverse(self.eye+alpha*TT+alpha*XX)).matmul(T)

    def compute(self, X, T, epoch, lr, device='cuda'):
        alpha = (X.shape[1]+T.shape[1]) / (X.shape[0]*self.eps)
        X = X - X.mean(dim=1).unsqueeze(1)
        X = X / X.norm(dim=1).unsqueeze(dim=1)
        XX = self.XX(X)

        self.eye = torch.eye(X.shape[0]).to(device)
        for i in tqdm(range(epoch)):
            T = T - T.mean(dim=1).unsqueeze(1)
            T = T / T.norm(dim=1).unsqueeze(dim=1)
            T += lr*self.gt(self.XX(T), XX, T, alpha)

        return T

class NNencoder(nn.Module):
    def __init__(self, act='ReLU'):
        super(NNencoder, self).__init__()
        self.l1 = nn.Linear(784, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 100)
        self.l4 = nn.Linear(100, 100)

        self.l5 = nn.Linear(100, 100)
        self.l6 = nn.Linear(100, 512)
        self.l7 = nn.Linear(512, 1024)
        self.l8 = nn.Linear(1024, 784)

        if act == 'ReLU':
            self.gate = nn.ReLU(inplace=True)
        elif act == 'Sigmoid':
            self.gate = nn.Sigmoid()
        elif act == 'tanh':
            self.gate = nn.Tanh()
        elif act == 'gelu':
            self.gate = nn.GELU()
        elif act == 'lrelu':
            self.gate = nn.LeakyReLU()
        else:
            raise ValueError

    def forward(self, x, recover=False, device='cpu'):
        rate = []
        m = x.shape[0]
        x = x.view(-1, 784)  # Flattern the (n,3,32,32) to (n,3096)

        x = self.gate(self.l1(x))
        x = self.gate(self.l2(x))
        x = self.gate(self.l3(x))
        x = self.gate(self.l4(x))

        if recover:
            x = self.gate(self.l5(x))
            x = self.gate(self.l6(x))
            x = self.gate(self.l7(x))
            x = self.gate(self.l8(x))

        return x

if __name__ == '__main__':
    from utils.tsne import tsne
    import torchvision.transforms as transforms
    from utils import useData
    from utils.utils import generateNormalizedClusteredData as gnd
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # data = gnd(cluster=torch.randn(10,100), std=1, num=50).to('cuda')
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )
    train, _ = useData(mode="MNIST", transform=transform, PATH="E:\\thesis\dataset")
    loader = DataLoader(train, batch_size=512, shuffle=True)
    for data, label in loader:
        with torch.no_grad():
            T = tsne(data.reshape(data.shape[0],-1).to('cuda'), 2, 50, 20.0)
        plt.scatter(T[:, 0].cpu(), T[:, 1].cpu(), 20, label.cpu())
        plt.show()
        break

    # for sample, target in loader:
    #     T = torch.randn(1000, 100).to('cuda')
    #     ae = autoencoder(0.05)
    #     T = ae.compute(
    #         X=sample.view(1000,-1).to('cuda'),
    #         T=T,
    #         epoch=1000,
    #         lr=0.1,
    #         device='cuda'
    #     )
    #     with torch.no_grad():
    #         T = tsne(T, 2, 50, 20.0)
    #         # data = tsne(sample.view(500,-1).to('cuda'), 2, 50, 20.0)
    #
    #     # plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), 20, target.cpu())
    #     # plt.show()
    #     plt.scatter(T[:, 0].cpu(), T[:, 1].cpu(), 20, target.cpu())
    #     plt.show()
    #
    #     break

    # with torch.no_grad():
    #     s = tsne(y, 2, 50, 20.0)
    #
    # plt.scatter(s[:, 0].cpu(), s[:, 1].cpu(), 20, label)
    # plt.show()


    # with torch.no_grad():
    #     data = tsne(data, 2, 50, 20.0)
    #
    # plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), 20, label)
    # plt.show()



