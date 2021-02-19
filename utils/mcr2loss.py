import numpy as np
import torch


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, device, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps
        self.device = device

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).to(self.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).to(self.device)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        X = X.reshape(X.shape[0], -1)
        W = X.T
        Pi = label_to_membership(Y.numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32).to(self.device)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi

        return (discrimn_loss_empi.item(), compress_loss_empi.item())
        # return (total_loss_empi,
        #         [discrimn_loss_empi.item(), compress_loss_empi.item()])
                # [discrimn_loss_theo.item(), compress_loss_theo.item()])

    # def forward(self, X):
    #     W = X.T
    #
    #     discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
    #     discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
    #
    #     return discrimn_loss_empi.item(), discrimn_loss_theo.item()

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.
    Parameters:
        targets (np.ndarray): matrix with one hot labels
    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)
    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

if __name__ == '__main__':
    from utils import rateReductionV2
    loss = MaximalCodingRateReduction()
    X = torch.randn(200, 3, 22, 22).view(200, -1).to('cuda:0') # M,C
    Y = torch.randint(low=0, high=10, size=(200,))
    X_bias = X + torch.randn_like(X) * 0.003
    X_bias = X_bias.to('cuda:0')

    print(loss(X, Y))


