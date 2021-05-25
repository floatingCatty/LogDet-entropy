import torch
import numpy as np

'''ON THE INFORMATION BOTTLENECK
THEORY OF DEEP LEARNING'''

def get_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = torch.square(X).sum(dim=1).unsqueeze(1)
    dists = x2 + x2.T - 2*X.matmul(X.T)
    return dists

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    N, dims = x.size()
    dists = get_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*np.log(2*np.pi*var)
    lprobs = torch.logsumexp(input=-dists2, dim=1) - np.log(N) - normconst
    h = - lprobs.mean()
    return dims/2 + h


def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    N, dims = x.size()
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = output.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)

if __name__ == '__main__':
    a = torch.randn(128,3)*0.3 + 0.7

    print(entropy_estimator_kl(a, var=0.1))


