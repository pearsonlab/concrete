import numpy as np
from numba import jit
from scipy.stats import multivariate_normal as multinorm


def em(y, K, step):
    """Implement the Baum-Welch algorithm.
    """
    T, D = y.shape

    # initialize transition parameters
    # A is a matrix of transition probabilities that acts to the left:
    # new_state = old_state @ A, so that rows of A sum to 1
    A = np.ones((K, K)) / K
    pi = np.ones(K) / K

    # initialize emission parameters
    mean = np.random.randn(K, D)
    cov = np.stack([np.diag(np.random.rand(D)) for _ in range(K)])

    """psi is the vector of evidence: p(y_t|z_t); it does not need to be
    normalized, but the lack of normalization will be reflected in logZ
    such that the end result using the given psi will be properly normalized
    when using the returned value of Z
    """
    psi = np.stack(
        [multinorm.pdf(y, mean[k], cov[k]) for k in range(K)], axis=-1)

    # initialize variables for e-step
    alpha = np.empty((T, K))  # p(z_t|y_{1:t})
    beta = np.empty((T, K))  # p(y_{t+1:T}|z_t) (unnormalized)
    gamma = np.empty((T, K))  # p(z_t|y_{1:T})
    logZ = np.empty(T)  # log partition function
    Xi = np.empty((T - 1, K, K))  # p(z_t, z_{t+1}|y_{1:T})

    for _ in range(step):
        a = psi[0] * pi
        alpha[0] = a / np.sum(a)
        logZ[0] = np.log(np.sum(a))
        b = np.ones(K)
        beta[-1, :] = b / K
        e_step(psi, A, alpha, beta, logZ, a, b, gamma, Xi)

        pi = gamma[0]
        m_step(y, gamma, Xi, mean, cov, A)
        psi = np.stack(
            [multinorm.pdf(y, mean[k], cov[k]) for k in range(K)], axis=-1)

    return A, pi, gamma, mean, cov


@jit(nopython=True)
def e_step(psi, A, alpha, beta, logZ, a, b, gamma, Xi):
    T = psi.shape[0]
    K = A.shape[0]

    # forward
    for t in range(1, T):
        asum = 0.0
        for i in range(K):
            a[i] = 0.0
            for j in range(K):
                a[i] += alpha[t - 1, j] * A[j, i] * psi[t, i]
            asum += a[i]

        for i in range(K):
            alpha[t, i] = a[i] / asum

        logZ[t] = np.log(asum)

    # backward
    for t in range(T - 1, 0, -1):
        bsum = 0.0
        for i in range(K):
            b[i] = 0.0
            for j in range(K):
                b[i] += beta[t, j] * A[i, j] * psi[t, j]
            bsum += b[i]

        for i in range(K):
            beta[t - 1, i] = b[i] / bsum

    # gamma
    for t in range(T):
        gamsum = 0.0
        for k in range(K):
            gamma[t, k] = alpha[t, k] * beta[t, k]
            gamsum += gamma[t, k]

        for k in range(K):
            gamma[t, k] /= gamsum

    if np.any(np.isnan(gamma)):
        raise ValueError('NaNs appear in posterior')

    # Xi
    for t in range(T - 1):
        xsum = 0.0
        for i in range(K):
            for j in range(K):
                Xi[t, i, j] = alpha[t, i] * A[i, j]
                Xi[t, i, j] *= beta[t + 1, j] * psi[t + 1, j]
                xsum += Xi[t, i, j]

        for i in range(K):
            for j in range(K):
                Xi[t, i, j] /= xsum

@jit(nopython=True)
def m_step(y, gamma, Xi, mean, cov, A):
    T, D = y.shape
    K = gamma.shape[1]

    for k in range(K):
        for d in range(D):
            ysum = 0.0
            gamsum = 0.0
            for t in range(T):
                ysum += gamma[t, k] * y[t, d]
                gamsum += gamma[t, k]
            mean[k, d] = ysum / gamsum

    for k in range(K):
        for i in range(D):
            for j in range(i, D):
                vsum = 0.0
                gamsum = 0.0
                for t in range(T):
                    vsum += gamma[t, k] * (y[t, i] - mean[k, i]) * (
                        y[t, j] - mean[k, j])
                    gamsum += gamma[t, k]
                cov[k, i, j] = vsum / gamsum
                cov[k, j, i] = cov[k, i, j]

    for i in range(K):
        asum = 0.0
        for j in range(K):
            for t in range(T - 1):
                A[i, j] += Xi[t, i, j]
            asum += A[i, j]

        for j in range(K):
            A[i, j] /= asum
