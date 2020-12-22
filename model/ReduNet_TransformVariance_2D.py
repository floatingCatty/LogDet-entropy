import cupy as np
from cupyx.scipy import signal
import scipy.special as sp
from utils import top_n, relu
from tqdm import tqdm
import torch.nn.functional as F
import torch

class ReduNet_2D(object):
    def __init__(self, e, nameta, n_class, in_channel, n_channel, kernel_size, L, lr, epsilon, adversial=False):
        '''
        Z is of size R(C, T, m) when type is 1D # complex data
             of size R(C, H, W, m) when type is 2D
        label is of size R(m)
        e: the maximal compression distortion rate
        nameta: the scale of prediction label Cz's variance
        k: the number of class
        l: the number of the model's layers
        lr: learning rate
        n_channel is the filter num to lifting the input Z into higher sparse dimension, making the final Channel num C*n_channel
        '''

        self.e = e
        self.nameta = nameta
        self.in_channel = in_channel
        self.n_channel = n_channel
        self.lr = lr
        self.n_class = n_class
        self.L = L
        self.adversial = adversial
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.random_filter = torch.randn(self.n_channel, self.in_channel, self.kernel_size[0], self.kernel_size[1])
        self.conv_grad = 0

    # def attack(self, X, label_X, image, convm):
    #     C, H, W, M = image.shape
    #     E_, C_, y = self._get_parameters_(V=X, label=label_X, mini_batch=-1)
    #     grad, _ = self.getGrad(V=X, E_=E_, C_=C_, y=y, mode=self.mode)
    #     # grad = np.sign(grad)
    #     # grad = grad / np.linalg.norm(grad.reshape(-1, grad.shape[-1]), axis=0).reshape((1, 1, 1, -1))
    #     grad = np.fft.fft2(grad, axes=(1, 2)).real
    #     grad = grad.reshape(self.n_channel, C, -1, M).transpose((1, 0, 2, 3))
    #     circ = np.array([[convm[i][j:] + convm[i][:j] for j in range(H * W)] for i in range(self.n_channel)])
    #
    #     grad = np.einsum('ckdm, kdt -> cktm', grad, circ).sum(axis=1).reshape(-1, H, W, M)
    #     image = image - self.epsilon * np.sign(grad)
    #
    #     return image.clip(0, 1)

    def dim_lift_sparce(self, X, kernel):
        X = torch.tensor(X.get()).permute(3,0,1,2)
        X.requires_grad = True
        output = F.conv2d(input=X, weight=kernel)
        # output.backward(X)
        #
        # self.conv_grad = X.grad

        output = output.permute(1,2,3,0).detach().numpy()

        return relu(np.array(output))


    def _sample_(self, V, label, mini_batch):
        ids = np.random.randint(0, V.shape[-1], (int(mini_batch * V.shape[-1])))

        return np.take(V, ids, axis=-1), label[ids]

    def _get_parameters_(self, V, label, mini_batch=-1):

        C = V.shape[0]
        m = V.shape[-1]

        # update PI
        if mini_batch > 0:
            V, label = self._sample_(V, label, mini_batch)
            m = V.shape[-1]
            PI = np.zeros((self.n_class, m), dtype=np.complex)
        else:
            PI = np.zeros((self.n_class, m), dtype=np.complex)

        a = C / (m * self.e ** 2)
        a_j = np.zeros(self.n_class)

        y = np.zeros(self.n_class)

        # update PI, gamma, alpha_j
        for j in range(self.n_class):
            PI[j] = np.array(np.equal(label, j), dtype=np.complex)

            a_j[j] = C / (PI[j].real.sum() * self.e ** 2)
            y[j] = PI[j].real.sum() / m

        C, H, W, m = V.shape
        E_ = a * np.linalg.inv(
            np.eye(C, dtype=np.complex)
            + a
            * np.einsum('khwm, chwm -> hwkc', V, V.conj())
        )

        # update C
        C_ = np.zeros((self.n_class, H, W, C, C), dtype=np.complex)
        for j in range(self.n_class):
            C_[j] = a_j[j] * np.linalg.inv(
                np.eye(C, dtype=np.complex)
                + a_j[j]
                * np.einsum('khwm,mn,chwn -> hwkc', V, np.diag(PI[j]), V.conj())
            )

        return E_, C_, y

    def getGrad(self, V, E_, C_, y):

        C, H, W, m = V.shape
        P = np.einsum('khwpc, chwm -> mkhwp', C_, V)
        norm = -self.nameta * np.linalg.norm(P.reshape(m, self.n_class, -1), axis=2)
        pi = np.array(sp.softmax(norm.get(), axis=1))

        grad = np.einsum('hwdc, ckwm -> dhwm', E_, V) \
               - np.einsum('mk, mkhwc -> chwm', y.reshape(1, -1) * pi, P)

        return grad, pi

    def _layer_(self, V, E_, C_, y, update_batchsize):

        index = 0
        m = V.shape[-1]

        layer_pi = np.zeros((m, self.n_class))

        while (index < m):
            # for each term, update the sample indiced by (index, index+update_batchsize)
            new_index = min(m, index + update_batchsize)

            grad, pi = self.getGrad(
                V=V[:, :, :, index:new_index],
                E_=E_,
                C_=C_,
                y=y
            )
            layer_pi[index:new_index] = pi
            V[:, :, :, index:new_index] += self.lr * grad

            index = new_index

        V = V / np.linalg.norm(V.reshape(-1, m), axis=0).reshape((1, 1, 1, -1))

        return V, layer_pi

    def get_loss(self, V, label):
        # V in shape(C, H, W, D)
        C, H, W, M = V.shape
        PI = np.zeros((self.n_class, M), dtype=np.complex)
        a = C / (M * self.e ** 2)
        a_j = np.zeros(self.n_class)
        y = np.zeros(self.n_class)

        for j in range(self.n_class):
            PI[j] = np.array(np.equal(label, j), dtype=np.complex)

            a_j[j] = C / (PI[j].real.sum() * self.e ** 2)
            y[j] = PI[j].real.sum() / M

        loss = np.log(np.linalg.det(a * np.einsum('thwm,chwm->hwtc', V, V.conj()) + np.eye(C)))
        for j in range(self.n_class):
            loss -= y[j] * np.log(
                np.linalg.det(a_j[j] * np.einsum('thwm,mn,chwn -> hwtc', V, np.diag(PI[j]), V.conj()) + np.eye(C)))
        loss = np.mean(loss)

        return loss

    def train(self, Z, label_Z, update_batchsize, mini_batch=-1, top_n_acc=1):
        '''
                mini_batch: the proportion of samples used to estimate the E and C
                update_batchsize: the number of samples containing in each batch when updating the new representation Z/V
                '''
        assert update_batchsize > 0
        assert 0 < mini_batch < 1 or mini_batch == -1

        acc = []
        # convert Z to V
        c, h, w, m = Z.shape
        Z = self._conv2d(Z, self.random_filter)
        Z = np.fft.fft2(Z, axes=(1, 2))  # R(C, H, W, m)
        Z = Z / np.linalg.norm(Z.reshape(-1, m), axis=0).reshape((1, 1, 1, -1))


        for _ in tqdm(range(self.L)):
            E_, C_, y = self._get_parameters_(V=Z, label=label_Z, mini_batch=mini_batch)
            Z, pi = self._layer_(
                V=Z,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize
            )

            acc.append(top_n(pre=pi, label=label_Z, n=top_n_acc))

        Z = np.fft.ifft2(Z, axes=(1, 2))

        return Z

    def estimate(self, Z, label_Z, X, label_X, update_batchsize, mini_batch=-1, top_n_acc=1):

        '''
                mini_batch: the proportion of samples used to estimate the E and C
                update_batchsize: the number of samples containing in each batch when updating the new representation Z/V
                '''

        assert update_batchsize > 0
        assert 0 < mini_batch < 1 or mini_batch == -1

        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []

        Z = self.dim_lift_sparce(Z, self.random_filter)
        Z = np.fft.fft2(Z, axes=(1, 2))  # R(C, H, W, m)
        Z = Z / np.linalg.norm(Z.reshape(-1, Z.shape[-1]), axis=0).reshape((1, 1, 1, -1))

        X_ = self.dim_lift_sparce(X, self.random_filter)
        X_ = np.fft.fft2(X_, axes=(1, 2))  # R(C, H, W, m)
        X_ = X_ / np.linalg.norm(X_.reshape(-1, X_.shape[-1]), axis=0).reshape((1, 1, 1, -1))

        # if self.adversial:
        #     X = self.attack(X_, label_X, X, self.random_filter)
        #     # np.save('data/mnist-atk_03.npy',X[:,:,:,:10])
        #     X_ = self._conv2d(X, self.random_filter)
        #     X_ = np.fft.fft2(X_, axes=(1, 2))  # R(C, H, W, m)
        #     X_ = X_ / np.linalg.norm(X_.reshape(-1, X_.shape[-1]), axis=0).reshape((1, 1, 1, -1))


        for _ in tqdm(range(self.L)):
            E_, C_, y = self._get_parameters_(V=Z, label=label_Z, mini_batch=mini_batch)
            Z, pi_Z = self._layer_(
                V=Z,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize
            )
            X_, pi_X = self._layer_(
                V=X_,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize
            )
            acc_train.append(top_n(pre=pi_Z, label=label_Z, n=top_n_acc))
            acc_test.append(top_n(pre=pi_X, label=label_X, n=top_n_acc))

            loss_train.append(self.get_loss(Z, label_Z).get())
            loss_test.append(self.get_loss(X_, label_X).get())

            print(acc_train[-1], acc_test[-1], loss_train[-1].real, loss_test[-1].real)


        X_ = np.fft.ifft2(X, axes=(1, 2))
        Z = np.fft.ifft2(Z, axes=(1, 2))

        return Z, X_, acc_train, acc_test, loss_train, loss_test
