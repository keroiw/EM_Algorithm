import pandas as pd


from generate_data import get_data
from em import init_em


if __name__ == "__main__":
    X = get_data()["X"]
    X = pd.DataFrame(X, columns=["T"+str(i) for i in range(1, X.shape[1]+1)])
    X_treatments = [X.loc[:, col] for col in X.columns]

    for X_t in X_treatments:
        init_em(X_t, 2)
