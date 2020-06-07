import numpy as np
import pandas as pd


from functools import reduce


class AlgorithmStateBase:
    __slots__ = ['theta_a', 'theta_b']

    def __init__(self, theta_a, theta_b):
        self.theta_a = theta_a
        self.theta_b = theta_b

    @staticmethod
    def compare(obj1, obj2, thresh=10e-2):
        theta_a1 = obj1.theta_a.flatten()
        theta_a2 = obj1.theta_a.flatten()
        theta_b1 = obj2.theta_b.flatten()
        theta_b2 = obj2.theta_b.flatten()
        theta_a_norm = np.linalg.norm(theta_a1 - theta_a2) < thresh
        theta_b_norm = np.linalg.norm(theta_b1 - theta_b2) < thresh
        return isinstance(obj1, AlgorithmStateInit) or all(np.array([theta_a_norm, theta_b_norm]) > thresh)


class AlgorithmStateInit(AlgorithmStateBase):

    def __init__(self, w):
        super().__init__(np.zeros((4, w)), np.zeros((4, w)))


class AlgorithmState(AlgorithmStateBase):

    def __init__(self, theta_a, theta_b):
        super().__init__(theta_a, theta_b)


def init_distr(w):
    theta = np.random.rand(4*w).reshape((4, -1))
    theta_sum = np.sum(theta, axis=0)
    return theta/theta_sum


def expectation(X: np.array, alpha, theta_a, theta_b):

    def get_cond_prob(X, theta):
        cond_prob = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x_i = X[i, :]
            x_i_prob = [theta[j-1, k] for k, j in zip(range(theta.shape[1]), x_i)]
            cond_prob[i] = reduce(lambda x, y: x * y, x_i_prob)
        return cond_prob

    cond_1 = get_cond_prob(X, theta_a)
    cond_2 = get_cond_prob(X, theta_b)
    weights_mtrx = np.vstack((cond_1, cond_2)).T
    alphas = np.array([alpha, 1 - alpha])
    cond_sums = weights_mtrx @ alphas.reshape((2, 1))
    weights_mtrx[:, 0] = (alpha * weights_mtrx[:, 0]) / cond_sums.flatten()
    weights_mtrx[:, 1] = ((1 - alpha) * weights_mtrx[:, 1]) / cond_sums.flatten()

    return weights_mtrx


def alpha_new(X, weights):
    N_k = np.sum(weights, axis=0)
    return 1-(N_k / np.sum(N_k))[0]


def new_theta(X_full, weights_full):

    def get_estimators(k):
        weights = weights_full[:, k]
        estimators = []
        for i in range(X_full.shape[1]):
            labels_pred = [X_full[:, i] == k for k in range(1, 5)]
            label_sums = np.array([np.sum(weights[pred]) for pred in labels_pred])
            estimators.append(label_sums / np.sum(weights))
        return np.array(estimators).T

    estimators_1 = get_estimators(0)
    estimators_2 = get_estimators(1)

    return estimators_1, estimators_2


def init_em(X: pd.Series, w: int):
    alpha = 0.5
    theta_a, theta_b = init_distr(w), init_distr(w)

    alg_state_old = AlgorithmStateInit(w)
    alg_state_new = AlgorithmStateInit(w)
    counter = 1
    while AlgorithmState.compare(alg_state_old, alg_state_new) and counter < 100:
        weights = expectation(X, alpha, theta_a, theta_b)
        # alpha = alpha_new(X, weights)
        theta_a, theta_b = new_theta(X, weights)

        alg_state_old = alg_state_new
        alg_state_new = AlgorithmState(theta_a, theta_b)
        counter += 1

    return theta_a, theta_b, alpha, counter