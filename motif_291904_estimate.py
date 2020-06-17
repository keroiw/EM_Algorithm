import json
import numpy as np
import argparse


from em_discrete import init_em


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input', default="./motify_test/gen_test.json", required=False,
                        help='Plik z danymi  (default: %(default)s)')
    parser.add_argument('--output', default="./motify_test/res_test.json", required=False,
                        help='Tutaj zapiszemy wyestymowane parametry  (default: %(default)s)')
    parser.add_argument('--estimate-alpha', default="no", required=False,
                        help='Czy estymowac alpha?  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


input_file, output_file, estimate_alpha = ParseArguments()

with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

alpha = data['alpha']
X = np.asarray(data['X'])
X = X.astype(int)

Theta, ThetaB, alpha, counter = init_em(X, 1000, alpha, estimate_alpha)
ThetaB = np.mean(ThetaB, 1)

estimated_params = {
    "alpha": alpha,
    "Theta": Theta.tolist(),
    "ThetaB": ThetaB.tolist()
}

with open(output_file, 'w') as outfile:
    json.dump(estimated_params, outfile)