import json
import numpy as np
import os
import random


params_file_path = os.path.join(os.getcwd(), 'params.json')


def get_data():
    with open(params_file_path, 'r') as inputfile:
        params = json.load(inputfile)

    w = params['w']
    k = params['k']
    alpha = params['alpha']

    # each column denotes parameters of particular distribution
    theta_a = np.array(params['Theta'])
    theta_b = np.array(params['ThetaB'])

    X = np.zeros((k, w))
    for i in range(k):
        distr = theta_a
        if random.random() < alpha:
            distr = theta_b
        X[i, :] = [np.random.choice([1, 2, 3, 4], size=1, p=distr[:, i]) for i in range(w)]

    return {
        "alpha": alpha,
        "X": X.tolist()
    }


