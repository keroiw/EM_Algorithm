import json
import numpy as np


w = 10
alpha = 0.5
k = 1000


theta_a = np.random.rand(w*4).reshape((4, 10))
theta_a = theta_a/np.sum(theta_a, axis=0)

# background distribution
# theta_b = np.random.rand(10*4).reshape((4, 10))
# theta_b = theta_b/np.sum(theta_b, axis=0)

theta_b = np.repeat(np.random.rand(4), w).reshape(4, -1)
theta_sum = np.sum(theta_b, axis=0)
theta_b = theta_b / theta_sum


params = {
    "w" : w,
    "alpha": alpha,
    "k": k,
    "Theta" : theta_a.tolist(),
    "ThetaB": theta_b.tolist()
    }


with open('params_simpl.json', 'w') as outfile:
    json.dump(params, outfile)
