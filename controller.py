import numpy as np
import scipy.stats as stats


from generate_data import get_data
# from em_gaussian import init_em
from em_discrete import init_em
from math import sqrt


def map_to_discr(distr: dict):
    pdf_vals = stats.norm(distr["mu"], sqrt(distr["var"])).pdf([1, 2, 3, 4])
    return pdf_vals / np.sum(pdf_vals)


if __name__ == "__main__":
    X = get_data()["X"].astype(int)
    theta_1, theta_2, alpha, counter = init_em(X, 3)
    print("Converged in: {0}".format(counter))
    print("Theta 1")
    print(np.round(theta_1, 3))
    print("Theta 2")
    print(np.round(theta_2, 3))
    print("Alpha")
    print(alpha)

