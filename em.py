import numpy as np
import pandas as pd
import random
import scipy.stats as stats

from math import sqrt


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
    sum_pdf = pdf_1*alpha + pdf_2*(1-alpha)
    w = np.zeros((X.size, len(dist_params)))
    w[:, 0] = pdf_1*alpha / sum_pdf
    w[:, 1] = pdf_2*(1 - alpha) / sum_pdf
    return w


def init_em(X: pd.Series, n_distr: int):
    labels = np.random.choice(list(range(1, n_distr + 1)), size=X.size)
    alpha = random.random()
    d1, d2 = init_distr_params()
    weights = expectation(X, alpha, [d1, d2])
    print()
