import numpy as onp
import jax.numpy as np
import matplotlib.pyplot as plt
from jax import lax
import ising_neq_v6_scan_functions_lax as ising

class TrajectorySimulator:
    def __init__(self, k1star, k2, k3, epsilon, J, N, transient_steps=1_000_000, max_steps=1_000_000):
        self.k1star = k1star
        self.k2 = k2
        self.k3 = k3
        self.epsilon = epsilon
        self.J = J
        self.N = N
        self.transient_steps = transient_steps
        self.max_steps = max_steps

        self.eb_list = onp.array([-0.1, 0, 0.1])
        self.t_trace_all = []
        self.m_trace_all = []

    def generate_trajectory(self, max_steps, k1, k2, k3, kn1, kn2, kn3, J, N, init_state=None):
        if init_state is None:
            init_state = (onp.random.choice([-1, 0, 1], size=(N, N)), 0)

        r1, r2 = onp.random.rand(max_steps), onp.random.rand(max_steps)
        r1, r2 = np.array(r1), np.array(r2)  # Convert to jax.numpy arrays

        def step_fn(carry, i):
            a, t = carry
            a, dt = ising.KMC_step(a, J, k1, k2, k3, kn1, kn2, kn3, N, r1=r1[i], r2=r2[i])
            t += dt
            return (a, t), (t, a.mean())

        final_state, (t_trace, m_trace) = lax.scan(
            step_fn, init_state, np.arange(max_steps)
        )
        return final_state, t_trace, m_trace


    def save_results(self):
        # Return results as a dict instead of list of lists
        results_dict = {}
        for i, (t, m) in enumerate(zip(self.t_trace_all, self.m_trace_all)):
            results_dict[f"t_trace_{i}"] = t
            results_dict[f"m_trace_{i}"] = m
        return results_dict
    
    def run_simulations(self):
        for eb in self.eb_list:
            params = {
                "k1": self.k1star * onp.exp(-eb),
                "k2": self.k2,
                "k3": self.k3,
                "kn1": self.k1star * self.epsilon * onp.exp(-eb),
                "kn2": self.k2 * self.epsilon,
                "kn3": self.k3 * self.epsilon,
                "J": self.J,
                "N": self.N,
            }
            thermal_state, _, _ = self.generate_trajectory(self.transient_steps, **params)
            final_state, t_trace, m_trace = self.generate_trajectory(
                self.max_steps, **params, init_state=(thermal_state[0], 0)
            )
            self.t_trace_all.append(t_trace)
            self.m_trace_all.append(m_trace)

    def plot_results(self, max_time_plot=800, save_path=None):
        fig = plt.figure(figsize=(8, 5), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, width_ratios=(6, 1), wspace=0.01, hspace=0.01)

        for i in range(3):
            if i == 0:
                ax1 = fig.add_subplot(gs[i, 0])
            else:
                ax1 = fig.add_subplot(gs[i, 0], sharex=ax1)
            ax2 = fig.add_subplot(gs[i, 1], sharey=ax1)
            ax1.plot(self.t_trace_all[i], self.m_trace_all[i], lw=2)

            m_hist, m_bins = onp.histogram(
                self.m_trace_all[i][1:],
                weights=onp.diff(self.t_trace_all[i]),
                bins=onp.linspace(-1, 1, 25),
                density=True,
            )
            ax2.stairs(m_hist, m_bins, fill=True, orientation="horizontal")

            ax2.yaxis.set_visible(False)
            ax1.xaxis.set_visible(False)
            ax1.set_ylabel(r"$a$")
            ax1.set_ylim(-1, 1)

            ax1.text(
                0,
                0.81,
                r"$\Delta E_B=$" + f"${self.eb_list[i]:.1f}$",
                transform=ax1.transAxes,
                fontsize=14,
            )

            a_mean = onp.sum(self.m_trace_all[i][1:] * onp.diff(self.t_trace_all[i])) / self.t_trace_all[i][-1]
            ax2.text(
                0,
                0.81,
                r"$\langle a \rangle=$" + f"${a_mean:.2f}$",
                transform=ax2.transAxes,
                fontsize=14,
            )

        ax1.xaxis.set_visible(True)
        ax1.set_xlabel(r"Time $t$")
        ax2.set_xlabel(r"PDF")
        ax1.set_xlim(0, max_time_plot)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
