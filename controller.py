import json
import numpy as np
import os
import scipy.stats as stats


from generate_data import get_data, get_thetas
from functools import reduce
from em_discrete import init_em
from math import sqrt


params_file_path = os.path.join(os.getcwd(), 'input_data', 'params.json')
# params_file_path = os.path.join(os.getcwd(), 'input_data', 'params_set1.json')


def map_to_discr(distr: dict):
    pdf_vals = stats.norm(distr["mu"], sqrt(distr["var"])).pdf([1, 2, 3, 4])
    return pdf_vals / np.sum(pdf_vals)


def single_run(X, max_rep=1000, verbose=False):
    theta_1, theta_2, alpha, counter = init_em(X, X.shape[1], max_rep)

    if verbose:
        print("Converged in: {0}".format(counter))
        print("Theta 1")
        print(np.round(theta_1, 3))
        print("Theta 2")
        print(np.round(theta_2, 3))
        print("Alpha")
        print(alpha)

    return theta_1, theta_2, alpha


def repeat_experiment(source_file: str, dest_file: str, reps=3, max_rep=1000):
    theta_a_lst = []
    theta_b_lst = []

    true_theta_a, true_theta_b = get_thetas(source_file)
    for i in range(reps):
        X = get_data(source_file)["X"].astype(int)
        theta_a, theta_b, alpha_1, counter = init_em(X, X.shape[1], max_rep)
        theta_a_lst.append(theta_a)
        theta_b_lst.append(theta_b)

    theta_a_est = reduce(lambda x, y: x + y, theta_a_lst) / len(theta_a_lst)
    theta_b_est = reduce(lambda x, y: x + y, theta_b_lst) / len(theta_b_lst)

    res_dict = {"theta_a_est": theta_a_est.tolist(),
                "theta_b_est": theta_b_est.tolist(),
                "theta_a": true_theta_a.tolist(),
                "theta_b": true_theta_b.tolist()}

    with open(dest_file, 'w') as outfile:
        json.dump(res_dict, outfile)

    print('Source: {0}'.format(source_file))
    print('Destination: {0}'.format(dest_file))


def read_results(source_file: str):
    with open(source_file, 'r') as input_file:
        params = json.load(input_file)
    return {k: np.array(v) for k, v in params.items()}


if __name__ == "__main__":
    # X = get_data(params_file_path)["X"].astype(int)
    # theta_1, theta_2, alpha = single_run(X, verbose=True)
    dest_file = os.path.join(os.getcwd(), 'saved_simulations', 'test.json')
    repeat_experiment(params_file_path, dest_file)
    read_results(dest_file)
