import cupy as np
from cupyx.scipy import signal
import scipy.special as sp
from utils import top_n, relu
from tqdm import tqdm
import torch.nn.functional as F
import torch

class ReduNet_2D(object):
    def __init__(self, e=0.1, nameta=1, n_class=10, in_channel=3, n_channel=5, kernel_size=(3,3), L=3000, lr=0.5):
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
        self.kernel_size = kernel_size
        self.random_filter = torch.randn(self.n_channel, self.in_channel, self.kernel_size[0], self.kernel_size[1])
        self.conv_grad = 0
        self.label_Z = []
        self.label_X = []


    def dim_lift_sparce(self, X, kernel):
        X = torch.tensor(X.get()).permute(3,0,1,2)
        X.requires_grad = True
        output = F.conv2d(input=X, weight=kernel)
        # output.backward(X)
        #
        # self.conv_grad = X.grad

        output = output.permute(1,2,3,0).detach().numpy()

        return relu(np.array(output))

    def update_label(self, label, pi, known):
        # update label using pi in R(m, n_class)
        new_label = label.copy()
        new_label[known:] = np.argmax(pi, axis=1)[known:]

        return new_label

    def _get_parameters_(self, V, label):
        '''
        in semi-supervised setting, proportion * m 's V has label
        '''
        C = V.shape[0]
        m = V.shape[-1]
        # update label using pi in R(m, n_class)

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

    def _layer_(self, V_, E_, C_, y, update_batchsize):

        V = V_.copy()
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
        # record the Rc and R along with Delta(R) separately.
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

    def estimate(self, Z, label_Z, X, label_X, update_batchsize, label_proportion=-1, top_n_acc=1):

        '''
                mini_batch: the proportion of samples used to estimate the E and C
                update_batchsize: the number of samples containing in each batch when updating the new representation Z/V
                '''

        assert update_batchsize > 0
        assert 0 < label_proportion < 1 or label_proportion == -1
        known = int(Z.shape[-1] * label_proportion)

        self.label_X = label_X.copy()
        self.label_Z = label_Z.copy()

        acc_train = []
        acc_test = []
        loss_train = []
        loss_test = []

        Z = self.dim_lift_sparce(Z, self.random_filter)
        Z = np.fft.fft2(Z, axes=(1, 2))  # R(C, H, W, m)
        Z = Z / np.linalg.norm(Z.reshape(-1, Z.shape[-1]), axis=0).reshape((1, 1, 1, -1))

        X = self.dim_lift_sparce(X, self.random_filter)
        X = np.fft.fft2(X, axes=(1, 2))  # R(C, H, W, m)
        X = X / np.linalg.norm(X.reshape(-1, X.shape[-1]), axis=0).reshape((1, 1, 1, -1))

        # init label using the proportional labels

        E_, C_, y = self._get_parameters_(V=Z[:,:,:,:known], label=label_Z[:known])
        _, pi_Z = self._layer_(
            V_=Z,
            E_=E_,
            C_=C_,
            y=y,
            update_batchsize=update_batchsize
        )

        label_Z = self.update_label(label=label_Z, pi=pi_Z, known=known)

        # iterate to get a stable label
        # while(True):
        #     E_, C_, y = self._get_parameters_(V=Z, label=label_Z)
        #     _, pi_Z = self._layer_(
        #         V_=Z,
        #         E_=E_,
        #         C_=C_,
        #         y=y,
        #         update_batchsize=update_batchsize
        #     )
        #     new_label_Z = self.update_label(label=label_Z, pi=pi_Z, known=known)
        #     print(np.equal(new_label_Z, label_Z).sum())
        #     if np.equal(new_label_Z, label_Z).sum() > 0.99*len(label_Z):
        #         label_Z = new_label_Z.copy()
        #         break
        #     label_Z = new_label_Z.copy()

        for _ in tqdm(range(self.L)):
            _, C_, y = self._get_parameters_(V=Z[:,:,:,:known], label=label_Z[:known])
            E_, _, _ = self._get_parameters_(V=Z, label=label_Z)
            Z, pi_Z = self._layer_(
                V_=Z,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize
            )
            X, pi_X = self._layer_(
                V_=X,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize
            )

            # E_, C_, y = self._get_parameters_(V=Z[:,:,:,:known], label=label_Z[:known])
            # _, pi_Z = self._layer_(
            #     V_=Z,
            #     E_=E_,
            #     C_=C_,
            #     y=y,
            #     update_batchsize=update_batchsize
            # )
            #
            # label_Z = self.update_label(label=label_Z, pi=pi_Z, known=known)
            #
            # while (True):
            #     E_, C_, y = self._get_parameters_(V=Z, label=label_Z)
            #     _, pi_Z = self._layer_(
            #         V_=Z,
            #         E_=E_,
            #         C_=C_,
            #         y=y,
            #         update_batchsize=update_batchsize
            #     )
            #     new_label_Z = self.update_label(label=label_Z, pi=pi_Z, known=known)
            #
            #     if np.equal(new_label_Z, label_Z).sum() > 0.99 * len(label_Z):
            #         label_Z = new_label_Z.copy()
            #         break
            #     label_Z = new_label_Z.copy()


            acc_train.append(top_n(pre=pi_Z, label=self.label_Z, n=top_n_acc))
            acc_test.append(top_n(pre=pi_X, label=self.label_X, n=top_n_acc))

            loss_train.append(self.get_loss(Z, self.label_Z).get())
            loss_test.append(self.get_loss(X, self.label_X).get())

            print(acc_train[-1], acc_test[-1], loss_train[-1].real, loss_test[-1].real)


        X_ = np.fft.ifft2(X, axes=(1, 2))
        Z = np.fft.ifft2(Z, axes=(1, 2))

        return Z, X_, acc_train, acc_test, loss_train, loss_test
