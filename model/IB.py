import torch
from tqdm import tqdm

class IB:
    def __init__(self, eps):
        self.eps = eps**2

    def XX(self, X):
        return X.matmul(X.T)

    def fixG(self, TT,XX,T,alpha):
        return alpha * (
                    torch.inverse(self.eye + alpha * TT) - torch.inverse(self.eye + alpha * TT + alpha * XX)).matmul(T)

    def compressionG(self, TT, XX, YY, T, alpha):
        return  alpha*(torch.inverse(self.eye+alpha*TT+alpha*XX)-torch.inverse(self.eye+alpha*TT+alpha*YY)).matmul(T)

    def evaluate(self, X_train, X_test, T, Y_train, fittingepoch,comepoch, lr, device='cuda'):
        alpha = (X_train.shape[1] + T.shape[1]) / (X_train.shape[0] * self.eps)
        X_train = (X_train - X_train.mean(dim=1).unsqueeze(1)) / X_train.norm(dim=1).unsqueeze(dim=1)
        X_test = (X_test - X_test.mean(dim=1).unsqueeze(1)) / X_test.norm(dim=1).unsqueeze(dim=1)
        Y_train = (Y_train - Y_train.mean(dim=1).unsqueeze(1)) / Y_train.norm(dim=1).unsqueeze(dim=1)
        XX_train = self.XX(X_train)
        XX_test = self.XX(X_test)
        YY_train = self.XX(Y_train)

        self.eye = torch.eye(X_train.shape[0]).to(device).float()
        # T_train fitting X_train
        T_train = T
        for _ in tqdm(range(fittingepoch)):
            T_train = (T_train - T_train.mean(dim=1).unsqueeze(1)) / (T_train.norm(dim=1).unsqueeze(dim=1)+1e-8)
            T_train += lr * self.fixG(self.XX(T_train), XX_train, T_train, alpha)

        # T fitting X_test
        for _ in tqdm(range(fittingepoch)):
            T = (T - T.mean(dim=1).unsqueeze(1)) / (T.norm(dim=1).unsqueeze(dim=1)+1e-8)
            T += lr * self.fixG(self.XX(T), XX_test, T, alpha)

        # T remove X by x_train T_train and Y_train
        for _ in tqdm(range(comepoch)):
            T = (T - T.mean(dim=1).unsqueeze(1)) / (T.norm(dim=1).unsqueeze(dim=1)+1e-8)
            T_train = (T_train - T_train.mean(dim=1).unsqueeze(1)) / (T_train.norm(dim=1).unsqueeze(dim=1)+1e-8)
            T += lr * self.compressionG(self.XX(T_train), XX_train, YY_train, T, alpha)
            T_train += lr * self.compressionG(self.XX(T_train), XX_train, YY_train, T_train, alpha)

        T = (T - T.mean(dim=1).unsqueeze(1)) / (T.norm(dim=1).unsqueeze(dim=1) + 1e-8)
        T_train = (T_train - T_train.mean(dim=1).unsqueeze(1)) / (T_train.norm(dim=1).unsqueeze(dim=1) + 1e-8)
        return T, T_train

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
    loader = DataLoader(train, batch_size=1000, shuffle=True)
    for sample, target in loader:
        Pi = torch.zeros(10, 500)
        for j in range(500):
            k = target[j]
            Pi[k, j] = 1.
        Pi = Pi.to('cuda').float()


        ae = IB(0.05)
        T, T_train = ae.evaluate(
            X_train=sample[:500].view(500,-1).to('cuda'),
            X_test=sample[500:].view(500,-1).to('cuda'),
            T=torch.randn(500,50).to('cuda'),
            Y_train=Pi.T,
            fittingepoch=500,
            comepoch=100,
            lr=0.1,
            device='cuda'
        )
        print(T)
        with torch.no_grad():
            T = tsne(T, 2, 50, 20.0)
        plt.scatter(T[:, 0].cpu(), T[:, 1].cpu(), 20, target[500:].cpu())
        plt.show()

        with torch.no_grad():
            T_train = tsne(T_train, 2, 50, 20.0)
        plt.scatter(T_train[:, 0].cpu(), T_train[:, 1].cpu(), 20, target[:500].cpu())
        plt.show()


