import numpy as np
import pandas as pd
import random
import scipy.stats as stats

from math import sqrt
from sklearn import mixture


class AlgorithmStateBase:
    __slots__ = ['alpha', 'mu', 'sigma']

    def __init__(self, alpha, mu, sigma):
        self.alpha = alpha
        self.mu = mu
        self.sigma = sigma

    @staticmethod
    def compare(obj1, obj2, thresh=10e-8):
        alpha_dist = abs(obj1.alpha - obj2.alpha)
        mu_dist = np.linalg.norm(obj1.mu - obj2.mu)
        sigma_dist = np.linalg.norm(obj1.sigma - obj2.sigma)
        return isinstance(obj1, AlgorithmStateInit) or np.all(np.array([alpha_dist, mu_dist, sigma_dist]) > thresh)


class AlgorithmStateInit(AlgorithmStateBase):

    def __init__(self):
        super().__init__(0, 0, 0)


class AlgorithmState(AlgorithmStateBase):

    def __init__(self, alpha, mu, sigma):
        super().__init__(alpha, mu, sigma)


def init_distr_params():
    d1 = {"mu": 1, "var": 2}
    d2 = {"mu": random.randint(0, 4), "var": random.random()}
    return d1, d2


def expectation(X: np.array, alpha: float, dist_params: list):
    # E part of an algorithm
    d1 = dist_params[0]
    d2 = dist_params[1]
    pdf_1 = stats.norm(d1["mu"], sqrt(d1["var"])).pdf(X)
    pdf_2 = stats.norm(d2["mu"], sqrt(d2["var"])).pdf(X)
    sum_pdf = pdf_1 * alpha + pdf_2 * (1 - alpha)
    w = np.zeros((X.size, len(dist_params)))
    w[:, 0] = pdf_1 * alpha / sum_pdf
    w[:, 1] = pdf_2 * (1 - alpha) / sum_pdf
    return w


def alpha_new(X, weights):
    N_k = np.sum(weights, axis=0)
    return (N_k / X.size)[0]


def mu_new(X, weights):
    N_k = np.sum(weights, axis=0)
    t = weights.T @ X
    return t / N_k


def sigma_new(X, mu, weights):
    N_k = np.sum(weights, axis=0)
    X_rep = np.vstack((X.values, X.values))
    t0 = X_rep - np.reshape(mu, (2, 1))
    t1 = (X_rep - np.reshape(mu, (2, 1))).T
    bias_squared = np.vstack((t0[0, :] @ t1[:, 0], t0[1, :] @ t1[:, 1]))
    weights_scaled = np.sum(weights.T * bias_squared, axis=1)
    return weights_scaled / N_k


def init_em(X: pd.Series):
    alpha = np.random.beta(a=2, b=2)
    d1, d2 = init_distr_params()

    alg_state_old = AlgorithmStateInit()
    alg_state_new = AlgorithmStateInit()
    counter = 1
    while AlgorithmState.compare(alg_state_old, alg_state_new):
        weights = expectation(X, alpha, [d1, d2])
        alpha = alpha_new(X, weights)
        mu_prim = mu_new(X, weights)
        sigma_prim = sigma_new(X, mu_prim, weights)

        d1["mu"] = mu_prim[0]
        d2["mu"] = mu_prim[1]
        d1["var"] = sigma_prim[0]
        d2["var"] = sigma_prim[1]

        alg_state_old = alg_state_new
        alg_state_new = AlgorithmState(alpha, mu_prim, sigma_prim)
        counter += 1

    # gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X.values.reshape(-1, 1))
    # gmm_weights = gmm.weights_.flatten()
    # gmm_mu = gmm.means_.flatten()
    # gmm_cov = gmm.covariances_.flatten()
    # return {"mu": gmm_mu[0], "var": gmm_cov[0]}, {"mu": gmm_mu[1], "var": gmm_cov[1]}, gmm_weights[0], 0

    return d1, d2, alpha, counter