import json
import numpy as np
import argparse
import random


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params', default="params_set1.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    parser.add_argument('--output', default="generated_data.json", required=False, help='Plik z Parametrami  (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output
    
    
param_file, output_file = ParseArguments()
 

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)
 

w = params['w']
k = params['k']
alpha = params['alpha']
theta_a = np.asarray(params['Theta'])
theta_b = np.asarray(params['ThetaB'])


X = np.zeros((k, w))
for i in range(k):
    distr = theta_a
    if random.random() < alpha:
        distr = theta_b
    X[i, :] = [np.random.choice([1, 2, 3, 4], size=1, p=distr[:, i]) for i in range(w)]


gen_data = {    
    "alpha": alpha,
    "X": X.tolist()
    }


with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
