import matplotlib.pyplot as plt
import seaborn as sns


def show_distributions(theta_a_est, theta_b_est, theta_a, theta_b):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    # cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    cmap = sns.color_palette("YlGnBu")

    def make_heatmap(ax, theta, x_ticks=True):
        sns.heatmap(theta,
                    ax=ax,
                    annot=True,
                    cmap=cmap,
                    cbar=False,
                    xticklabels=x_ticks)

    make_heatmap(ax1, theta_a_est, x_ticks=False)
    make_heatmap(ax3, theta_b_est)
    make_heatmap(ax2, theta_a, x_ticks=False)
    make_heatmap(ax4, theta_b)

    ax1.set_title('Estimated')
    ax2.set_title('Theoretical')

    plt.plot()


