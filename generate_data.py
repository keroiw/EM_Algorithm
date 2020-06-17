import json
import numpy as np
import random


def get_thetas(source_file: str):
    with open(source_file, 'r') as inputfile:
        params = json.load(inputfile)

    theta_a = np.array(params['Theta'])
    theta_b = np.array(params['ThetaB'])

    return theta_a, theta_b


def get_data(source_file: str):
    with open(source_file, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    k = params['k']
    alpha = params['alpha']

    # each column denotes parameters of particular distribution
    theta_a, theta_b = get_thetas(source_file)
    X = np.zeros((k, w))
    for i in range(k):
        distr = theta_a
        if random.random() > alpha:
            distr = theta_b
        X[i, :] = [np.random.choice([1, 2, 3, 4], size=1, p=distr[:, i]) for i in range(w)]


    return {
        "alpha": alpha,
        "X": X,
    }



