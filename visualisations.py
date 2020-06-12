import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def show_distributions(theta_a_est, theta_b_est, theta_a, theta_b):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    cmap = sns.color_palette("YlGnBu")

    def make_heatmap(ax, theta, x_ticks=True, y_ticks=True):
        theta = pd.DataFrame(theta)
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
