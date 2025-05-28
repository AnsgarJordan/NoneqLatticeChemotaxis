import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import os

import param_config
import ising_neq_v6_scan_functions_lax as ising
from results_saver import ResultsSaver

## set parameters
plt.rcParams.update(
    {
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
        "legend.fontsize": 16,  # this is the font size in legends
        "xtick.labelsize": 16,  # this and next are the font of ticks
        "ytick.labelsize": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 20,  # this is the foflags.N of axes labels
        "savefig.format": "pdf",  # how figures should be saved
        "legend.edgecolor": "0.0",
        "legend.framealpha": 0.0,
        # "text.usetex": True,
    }
)

plasmamap = plt.get_cmap("plasma")
RdBumap = plt.get_cmap("RdBu")

flags = param_config.parser.parse_args("")

if os.path.exists(flags.dataname):
    os.stat(flags.dataname)
else:
    os.makedirs(flags.dataname)

# Set parameters
N = flags.N
k2 = flags.k2
k3 = flags.k3
# epsilon = flags.epsilon
# DeltaG = -3 * np.log(epsilon)
J = flags.J
N = flags.N
k1star = 1.5946235656738281


args_evaluating = {
    "max_steps": 5, # flags.max_step_evaluating,
    "activity_only": False,
    "transient_steps": 5, # flags.step_thermalize,
    "dwell_threshold_factor": 1.5,
    "switching_threshold_factor": 1.5,
    "prominence_threshold": flags.prominence_threshold,
    "use_kde": True,
    "dataname": None,
    "random_seed": flags.random_seed,
    "m_thresholds": None,
    "return_thresholds": True,
}

### run the simulation
deltaG_array = np.load("fig2a_data.npz")["DeltaG"]
t_switching_mean = []
t_switching_std = []
t_switching_asymmetry_mean = []
t_switching_asymmetry_std = []

for deltaG in deltaG_array:

    epsilon = np.exp(-deltaG / 3)
    results = ising.calc_dwell_and_switching_times(
        k1star,
        k2,
        k3,
        k1star * epsilon,
        k2 * epsilon,
        k3 * epsilon,
        J,
        N,
        **args_evaluating,
    )
    t_up = np.array(results[1][0])     # switching up
    t_down = np.array(results[1][1])   # switching down

    # Combine up & down
    t_all = np.concatenate([t_up, t_down])
    t_switching_mean.append(np.mean(t_all))
    t_switching_std.append(np.std(t_all))

    # Asymmetry (log ratio of means)
    if len(t_up) > 0 and len(t_down) > 0:
        asymmetry = np.log(np.mean(t_down) / np.mean(t_up))
        t_switching_asymmetry_mean.append(asymmetry)
        # Optional: compute std of log ratios per sample (less common)
        asymmetry_std = np.std(np.log(t_down)) + np.std(np.log(t_up))  # very rough estimate
        t_switching_asymmetry_std.append(asymmetry_std)
    else:
        t_switching_asymmetry_mean.append(np.nan)
        t_switching_asymmetry_std.append(np.nan)

## make the data array
data = {
    "DeltaG":deltaG_array,
    "t_switching_mean":t_switching_mean,
    "t_switching_std":t_switching_std,
    "t_switching_asymmetry":t_switching_asymmetry_mean,
    "t_switching_asymmetry_std":t_switching_asymmetry_std
}
params = {
    "k1star": k1star,
    "k2": k2,
    "k3": k3,
    "epsilon": epsilon,
    "J": J,
    "N": N,
    **args_evaluating,
}
saver = ResultsSaver()
saver.save(data, params)


## Plotting section
fig, ax = plt.subplots(figsize=(6, 5))
ax.errorbar(
    data["DeltaG"],
    data["t_switching_mean"][0],
    yerr=data["t_switching_std"][0],
    fmt="o-",
    label=r"$\tau_+$",
    capsize=3,
    color="tab:red",
)
ax.errorbar(
    data["DeltaG"],
    data["t_switching_mean"][1],
    yerr=data["t_switching_std"][1],
    fmt="o-",
    label=r"$\tau_-$",
    capsize=3,
    color="tab:blue",
)

ax.set_xlabel(r"$\Delta G$")
ax.set_ylabel(r"Switching time $\tau_{\pm}$")
ax.legend()

fig.savefig("plots/fig2A_switching_times_vs_dG.png", dpi=300, bbox_inches="tight")
plt.close(fig)

## switching time
fig, ax = plt.subplots(figsize=(4, 3))
ax.errorbar(
    data["DeltaG"],
    data["t_switching_asymmetry"],
    yerr=data["t_switching_asymmetry_std"],
    fmt="o-",
    capsize=3,
    color="k",
)
ax.set_xlabel(r"$\Delta G$")
# ax.set_ylabel(r"Relative asymmetry")
ax.set_ylabel(r"$(\tau_--\tau_+)/\langle \tau\rangle$")
# ax.legend()


fig.savefig("plots/fig2A_switching_times_inset.png", dpi=300, bbox_inches="tight")
plt.close(fig)