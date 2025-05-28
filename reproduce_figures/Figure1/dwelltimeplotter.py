import numpy as np
import matplotlib.pyplot as plt

class DwellTimePlotter:
    def __init__(self, t_dwell_up, t_dwell_down, t_switch_up=None, t_switch_down=None):
        self.t_dwell_up = t_dwell_up
        self.t_dwell_down = t_dwell_down
        self.t_switch_up = t_switch_up
        self.t_switch_down = t_switch_down

    def plot_dwell_times(self, save_path=None):

        if self.t_dwell_up is None or self.t_dwell_down is None:
            raise ValueError("Dwell time data not provided")

        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        t_plot_max = (self.t_dwell_up.mean() + self.t_dwell_down.mean()) * 4
        bins = np.linspace(0, t_plot_max, 20)
        bins_center = (bins[1:] + bins[:-1]) / 2
        counts_up, _ = np.histogram(self.t_dwell_up, bins=bins, density=True)
        counts_down, _ = np.histogram(self.t_dwell_down, bins=bins, density=True)

        ax.plot(bins_center, counts_up, "-", label="$t_+$", color="tab:red", lw=2.5)
        ax.plot(bins_center, counts_down, "-", label="$t_-$", color="tab:blue", lw=2.5)
        ax.set_xlim(0, t_plot_max)
        ax.legend(frameon=False)
        ax.set_yscale("log")
        ax.set_xlabel(r"Dwell time $t_{\pm}$")
        ax.set_ylabel(r"Probability density")

        if save_path:
            plt.savefig(save_path)
            print(f"Dwell times plot saved to {save_path}")

    def plot_switching_times(self, save_path=None):
        if self.t_switch_up is None or self.t_switch_down is None:
            raise ValueError("Switching time data not provided")

        fig, ax = plt.subplots(figsize=(6, 5), tight_layout=True)
        t_switch_up_mean = self.t_switch_up.mean()
        t_switch_down_mean = self.t_switch_down.mean()
        t_plot_max = (t_switch_up_mean + t_switch_down_mean) * 2

        bins = np.linspace(0, t_plot_max, 35)
        counts_up, _ = np.histogram(self.t_switch_up, bins=bins, density=True)
        counts_down, _ = np.histogram(self.t_switch_down, bins=bins, density=True)
        bins_center = (bins[1:] + bins[:-1]) / 2

        ax.plot(bins_center, counts_up, "-", label=r"$\tau_{+}$", color="tab:red", lw=2.5)
        ax.plot(bins_center, counts_down, "-", label=r"$\tau_{-}$", color="tab:blue", lw=2.5)

        ax.axvline(t_switch_up_mean, color="tab:red", linestyle="--", lw=2.5, label=r"$\langle \tau_{+}\rangle$")
        ax.axvline(t_switch_down_mean, color="tab:blue", linestyle="--", lw=2.5, label=r"$\langle \tau_{-}\rangle$")

        y_arrow = (counts_down.max() + counts_up.max()) / 2
        ax.annotate(
            "",
            xy=(t_switch_up_mean, y_arrow),
            xytext=(t_switch_down_mean, y_arrow),
            arrowprops=dict(arrowstyle="<->", color="k", lw=2),
        )
        ax.text(
            (t_switch_up_mean + t_switch_down_mean) / 2,
            y_arrow + 0.01,
            r"$\Delta\tau$",
            ha="center",
            fontsize=18,
        )

        ax.legend(frameon=False)
        ax.set_ylim(0, max(counts_up.max(), counts_down.max()) * 1.1)
        ax.set_xlabel(r"Switching time $\tau_{\pm}$", fontsize=18)
        ax.set_ylabel(r"Probability density", fontsize=18)
        ax.set_xlim(bins[0], bins[-1])

        if save_path:
            plt.savefig(save_path)
            print(f"Switching times plot saved to {save_path}")

