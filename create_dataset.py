import json
import numpy as np


theta_a = np.random.rand(10*4).reshape((4, 10))
theta_a = theta_a/np.sum(theta_a, axis=0)

# background distribution
theta_b = np.random.rand(10*4).reshape((4, 10))
theta_b = theta_b/np.sum(theta_b, axis=0)

params = {
    "w" : 10,
    "alpha" : 0.5,
    "k" : 10000,
    "Theta" : theta_a.tolist(),
    "ThetaB" : theta_b.tolist()
    }


with open('params_set1.json', 'w') as outfile:
    json.dump(params, outfile)
