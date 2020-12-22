import cupy as np
from cupyx.scipy import signal
import scipy.special as sp
from utils import top_n, relu
from tqdm import tqdm

class ReduNet_1D(object):
    def __init__(self, e, nameta, k, n_channel, kernel_size, L, lr, epsilon, mode='1D', adversial=False):
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
        self.n_channel = n_channel
        self.lr = lr
        self.k = k
        self.L = L
        self.mode = mode
        self.adversial = adversial
        self.kernel_size = kernel_size
        self.epsilon = epsilon
        self.random_filter = np.random.randn(self.n_channel, self.kernel_size[0], self.kernel_size[1])

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


    def liftingAndSparseCoding(self, X, convm, k):
        # X in size(C, D, m)
        C, D, m = X.shape

        circ = np.array([[convm[i][j:] + convm[i][:j] for j in range(D)] for i in range(k)])

        return relu(np.einsum('ktd,cdm->kctm', circ, X).reshape((-1, D, m)))

    def _conv2d(self, X, kernel):
        k, w, h = kernel.shape
        C, H, W, m = X.shape
        new_X = np.zeros((C*k, H, W, m))
        for i in range(m):
            new_X[:,:,:,i] = \
                np.concatenate(
                (
                    np.concatenate(
                        (signal.convolve2d(X[j, :, :, i], kernel[o], mode='same') for
                         o in range(k)),
                        axis=0
                    ).reshape(k, H, W) for j in range(C)
                ),
                axis=0
            )
        return relu(new_X)



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
            PI = np.zeros((self.k, m), dtype=np.complex)
        else:
            PI = np.zeros((self.k, m), dtype=np.complex)

        a = C / (m * self.e ** 2)
        a_j = np.zeros(self.k)

        y = np.zeros(self.k)

        # update PI, gamma, alpha_j
        for j in range(self.k):
            PI[j] = np.array(np.equal(label, j), dtype=np.complex)

            a_j[j] = C / (PI[j].real.sum() * self.e ** 2)
            y[j] = PI[j].real.sum() / m

        # update E
        if self.mode is '1D':
            C, T, m = V.shape
            E_ = a * np.linalg.inv(
                np.eye(C)
                + a
                * np.einsum('ktm, ctm -> tkc', V, V.conj())
            )

            C_ = np.zeros((self.k, T, C, C), dtype=np.complex)
            # update C
            for j in range(self.k):
                C_[j] = a_j[j] * np.linalg.inv(
                    np.eye(C, dtype=np.complex)
                    + a_j[j]
                    * np.einsum('ktm,mn,ctn -> tkc', V, np.diag(PI[j]), V.conj())
                )

        elif self.mode is '2D':
            C, H, W, m = V.shape
            E_ = a * np.linalg.inv(
                np.eye(C, dtype=np.complex)
                + a
                * np.einsum('khwm, chwm -> hwkc', V, V.conj())
            )

            # update C
            C_ = np.zeros((self.k, H, W, C, C), dtype=np.complex)
            for j in range(self.k):
                C_[j] = a_j[j] * np.linalg.inv(
                    np.eye(C, dtype=np.complex)
                    + a_j[j]
                    * np.einsum('khwm,mn,chwn -> hwkc', V, np.diag(PI[j]), V.conj())
                )

        return E_, C_, y

    def getGrad(self, V, E_, C_, y, mode='2D'):
        if mode is '1D':
            C, T, m = V.shape
            P = np.einsum('ktpc, ctm -> mktp', C_, V)
            norm = -self.nameta * np.linalg.norm(P, axis=(2, 3))
            pi = np.array(sp.softmax(norm.get(), axis=1))

            grad = np.einsum('tdc, ctm -> dtm', E_, V) \
                   - np.einsum('mk, mktc -> ctm', y.reshape(1, -1) * pi, P)

        elif mode is '2D':

            C, H, W, m = V.shape
            P = np.einsum('khwpc, chwm -> mkhwp', C_, V)
            norm = -self.nameta * np.linalg.norm(P.reshape(m, self.k, -1), axis=2)
            pi = np.array(sp.softmax(norm.get(), axis=1))

            grad = np.einsum('hwdc, ckwm -> dhwm', E_, V) \
                   - np.einsum('mk, mkhwc -> chwm', y.reshape(1, -1) * pi, P)

        else:
            raise ValueError

        return grad, pi

    def _layer_(self, V, E_, C_, y, update_batchsize, mode='2D'):

        index = 0
        m = V.shape[-1]

        layer_pi = np.zeros((m, self.k))

        if mode is '1D':
            while (index < m):
                # for each term, update the sample indiced by (index, index+update_batchsize)
                new_index = min(m, index + update_batchsize)

                grad, pi = self.getGrad(
                    V=V[:, :, index:new_index],
                    E_=E_,
                    C_=C_,
                    y=y,
                    mode=mode
                )

                layer_pi[index:new_index] = pi
                V[:, :, index:new_index] += self.lr * grad

                index = new_index

        elif mode is '2D':
            while (index < m):
                # for each term, update the sample indiced by (index, index+update_batchsize)
                new_index = min(m, index + update_batchsize)

                grad, pi = self.getGrad(
                    V=V[:, :, :, index:new_index],
                    E_=E_,
                    C_=C_,
                    y=y,
                    mode=mode
                )
                layer_pi[index:new_index] = pi
                V[:, :, :, index:new_index] += self.lr * grad

                index = new_index
        else:
            raise ValueError

        # normalize V
        if mode is '1D':
            V = V / np.linalg.norm(V, axis=(0, 1)).reshape((1, 1, -1))
        else:
            V = V / np.linalg.norm(V.reshape(-1, m), axis=0).reshape((1, 1, 1, -1))

        return V, layer_pi

    def get_loss(self, V, label):
        # V in shape(C, H, W, D)
        C, H, W, M = V.shape
        PI = np.zeros((self.k, M), dtype=np.complex)
        a = C / (M * self.e ** 2)
        a_j = np.zeros(self.k)
        y = np.zeros(self.k)

        for j in range(self.k):
            PI[j] = np.array(np.equal(label, j), dtype=np.complex)

            a_j[j] = C / (PI[j].real.sum() * self.e ** 2)
            y[j] = PI[j].real.sum() / M

        loss = np.log(np.linalg.det(a * np.einsum('thwm,chwm->hwtc', V, V.conj()) + np.eye(C)))
        for j in range(self.k):
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
        if self.mode is '1D':
            Z = self.liftingAndSparseCoding(Z, self.random_filter, self.n_channel)
            Z = np.fft.fft(Z, axis=1)
            Z = Z / np.linalg.norm(Z, axis=(0, 1)).reshape((1, 1, Z.shape[-1]))
        elif self.mode is '2D':
            c, h, w, m = Z.shape
            Z = self.liftingAndSparseCoding(Z.reshape(c, -1, m), self.random_filter, self.n_channel).reshape(-1, h, w, m)
            Z = np.fft.fft2(Z, axes=(1, 2))  # R(C, H, W, m)
            Z = Z / np.linalg.norm(Z.reshape(-1, m), axis=0).reshape((1, 1, 1, -1))
        else:
            raise ValueError

        for _ in tqdm(range(self.L)):
            E_, C_, y = self._get_parameters_(V=Z, label=label_Z, mini_batch=mini_batch)
            Z, pi = self._layer_(
                V=Z,
                E_=E_,
                C_=C_,
                y=y,
                update_batchsize=update_batchsize,
                mode=self.mode
            )

            acc.append(top_n(pre=pi, label=label_Z, n=top_n_acc))

        if self.mode is '1D':
            Z = np.fft.ifft(Z, axis=1)
        else:
            Z = np.fft.ifft2(Z, axes=(1, 2))

        return Z

    def estimate(self, Z, label_Z, X, label_X, update_batchsize, mini_batch=-1, top_n_acc=1):

        '''
                mini_batch: the proportion of samples used to estimate the E and C
                update_batchsize: the number of samples containing in each batch when updating the new representation Z/V
                '''

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

        if self.mode is '1D':
            Z = self.liftingAndSparseCoding(Z, self.random_filter, self.n_channel)
            X = self.liftingAndSparseCoding(X, self.random_filter, self.n_channel)
            Z = np.fft.fft(Z, axis=1)
            Z = Z / np.linalg.norm(Z, axis=(0, 1)).reshape((1, 1, Z.shape[-1]))
            X = np.fft.fft(X, axis=1)
            X = X / np.linalg.norm(X, axis=(0, 1)).reshape((1, 1, X.shape[-1]))

        elif self.mode is '2D':
            c, h, w, m = Z.shape
            C, H, W, M = X.shape
            Z = self._conv2d(Z, self.random_filter)
            Z = np.fft.fft2(Z, axes=(1, 2))  # R(C, H, W, m)
            Z = Z / np.linalg.norm(Z.reshape(-1, Z.shape[-1]), axis=0).reshape((1, 1, 1, -1))

            X_ = self._conv2d(X, self.random_filter)
            # X_ = self.liftingAndSparseCoding(X.reshape(C, -1, M), self.random_filter, self.n_channel).reshape(-1, H, W, M)
            X_ = np.fft.fft2(X_, axes=(1, 2))  # R(C, H, W, m)
            X_ = X_ / np.linalg.norm(X_.reshape(-1, X_.shape[-1]), axis=0).reshape((1, 1, 1, -1))

            if self.adversial:
                X = self.attack(X_, label_X, X, self.random_filter)
                # np.save('data/mnist-atk_03.npy',X[:,:,:,:10])
                X_ = self.liftingAndSparseCoding(X.reshape(C, -1, M), self.random_filter, self.n_channel).reshape(-1, H,
                                                                                                                 W, M)
                X_ = np.fft.fft2(X_, axes=(1, 2))  # R(C, H, W, m)
                X_ = X_ / np.linalg.norm(X_.reshape(-1, X_.shape[-1]), axis=0).reshape((1, 1, 1, -1))

        else:
            raise ValueError

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
            print(acc_train[-1], acc_test[-1])

            loss_train.append(self.get_loss(Z, label_Z).get())
            loss_test.append(self.get_loss(X_, label_X).get())

        if self.mode is '1D':
            X_ = np.fft.ifft(X, axis=1)
            Z = np.fft.ifft(Z, axis=1)
        else:
            X_ = np.fft.ifft2(X, axes=(1, 2))
            Z = np.fft.ifft2(Z, axes=(1, 2))

        return Z, X_, acc_train, acc_test, loss_train, loss_test


