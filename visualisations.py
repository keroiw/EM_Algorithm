import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from em_discrete import init_em_val


def show_distributions(theta_a_est, theta_b_est, theta_a, theta_b):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 5))
    cmap = sns.color_palette("YlGnBu")

    def make_heatmap(ax, theta, x_ticks=True, y_ticks=True):
        theta = pd.DataFrame(theta).round(3)
        theta["Support"] = [str(i) for i in range(1, 5)]
        theta.set_index("Support", inplace=True)
        heatmap = sns.heatmap(theta,
                              ax=ax,
                              annot=True,
                              cmap=cmap,
                              cbar=False,
                              xticklabels=x_ticks,
                              yticklabels=y_ticks)
        if not y_ticks:
            heatmap.set_ylabel('')

    make_heatmap(ax1, theta_a_est, x_ticks=False)
    make_heatmap(ax3, theta_b_est)
    make_heatmap(ax2, theta_a, x_ticks=False, y_ticks=False)
    make_heatmap(ax4, theta_b, y_ticks=False)

    ax1.set_title('Estimated')
    ax2.set_title('Theoretical')

    plt.plot()


def plot_convergence(X, theta_a_org, theta_b_org, max_rep=1000, est_alpha="no", alpha=0.5):

    counter, norms_a, norms_b, alpha_lst = init_em_val(X, max_rep,  X.shape[1], est_alpha, alpha, theta_a_org, theta_b_org)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    norms = [np.array(n) for n in (norms_a, norms_b, alpha_lst)]
    norms = np.vstack(norms).T
    norms = pd.DataFrame(norms, columns=["ThetaA", "ThetaB", "Alpha"])
    norms["iterations"] = np.array(list(range(1, counter)))

    fig.suptitle("Estimators difference in subsequent iterations", fontsize=14)
    sns.lineplot(x="Iterations", y="ThetaA", data=norms, ax=ax1)
    ax1.set(ylabel=r'$\Vert\hat\Theta^A - \Theta^A\Vert_2$')

    sns.lineplot(x="Iteration", y="ThetaB", data=norms, ax=ax2)
    ax2.set(ylabel=r'$\Vert\hat\Theta^B - \Theta^B\Vert_2$')

    plt.show()
