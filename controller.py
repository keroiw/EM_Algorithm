import numpy as np
import pandas as pd
import scipy.stats as stats


from generate_data import get_data
from em import init_em
from math import sqrt


def map_to_discr(distr: dict):
    pdf_vals = stats.norm(distr["mu"], sqrt(distr["var"])).pdf([1, 2, 3, 4])
    return pdf_vals / np.sum(pdf_vals)


if __name__ == "__main__":
    X = get_data()["X"]
    X = pd.DataFrame(X, columns=["T"+str(i) for i in range(1, X.shape[1]+1)])
    X_treatments = [X.loc[:, col] for col in X.columns]

    theta_1 = np.zeros((4, X.shape[1]))
    theta_2 = np.zeros((4, X.shape[1]))
    alphas = np.zeros(4,)
    for i, X_t in enumerate(X_treatments):
        d1, d2, alpha = init_em(X_t)
        theta_1[:, i] = map_to_discr(d1)
        theta_2[:, i] = map_to_discr(d2)
        alphas[i] = alpha

