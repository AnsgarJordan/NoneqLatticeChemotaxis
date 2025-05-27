from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import lax
import numpy as onp
import matplotlib.pyplot as plt
import argparse, os

import ising_neq_v6_scan_functions_lax as ising
import param_config
from dwelltimeplotter import DwellTimePlotter
from results_saver import ResultsSaver
from trajectorysimulator import TrajectorySimulator

plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "legend.fontsize": 14,  # this is the font size in legends
        "xtick.labelsize": 14,  # this and next are the font of ticks
        "ytick.labelsize": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 18,  # this is the foflags.N of axes labels
        "savefig.format": "pdf",  # how figures should be saved
        "legend.edgecolor": "0.0",
        "legend.framealpha": 1.0,
    }
)

plasmamap = plt.get_cmap("plasma")

flags = param_config.parser.parse_args("")

if os.path.exists(flags.dataname):
    os.stat(flags.dataname)
else:
    os.makedirs(flags.dataname)

# Set parameters
N = flags.N
k2 = flags.k2
k3 = flags.k3
epsilon = flags.epsilon
DeltaG = -3 * np.log(epsilon)
J = flags.J
N = flags.N
k1star = 1.5946235656738281

args_balancing_simu = {
    "transient_steps": flags.step_thermalize,
    "dwell_threshold_factor": 1.5,
    "switching_threshold_factor": 1.5,
    "prominence_threshold": flags.prominence_threshold,
    "use_kde": True,
    "dataname": None,
    "return_thresholds": False,
}
args_balancing_overall = {
    "max_steps": flags.max_step_balancing,
    "activity_tolerance": flags.activity_threshold,
    "k1_tolerance": 1e-3,
    "params": args_balancing_simu,
}

# Step 1: we calculate k1star to make sure we have a bistable system for the lattice size, energy, etc that we have chosen 
# k1star = ising.calc_kstar_binary(
#     k2, k3, k2 * epsilon, k3 * epsilon, J, epsilon, N, **args_balancing_overall
# )

args_evaluating = {
    "max_steps": flags.max_step_evaluating,
    "activity_only": False,
    "transient_steps": flags.step_thermalize,
    "dwell_threshold_factor": 1.5,
    "switching_threshold_factor": 1.5,
    "prominence_threshold": flags.prominence_threshold,
    "use_kde": True,
    "dataname": None,
    "random_seed": flags.random_seed,
    "m_thresholds": None,
    "return_thresholds": True,
}

# # Step 2: Run the simulation for the chosen k1star and obtain dwell/switch times  
# results = ising.calc_dwell_and_switching_times(
#     k1star,
#     k2,
#     k3,
#     k1star * epsilon,
#     k2 * epsilon,
#     k3 * epsilon,
#     J,
#     N,
#     **args_evaluating,
# )

# # save all of the parameters to a file 
# params = {
#     "k1star": k1star,
#     "k2": k2,
#     "k3": k3,
#     "epsilon": epsilon,
#     "J": J,
#     "N": N,
#     **args_evaluating,
# }

# saver = ResultsSaver()
# saver.save(results, params)

# t_dwell_up, t_dwell_down = onp.array(results[0][0]), onp.array(results[0][1])
# t_switch_up, t_switch_down = onp.array(results[1][0]), onp.array(results[1][1])

# # plot the dwell times etc 
# plotter = DwellTimePlotter(t_dwell_up, t_dwell_down, t_switch_up, t_switch_down)
# plotter.plot_dwell_times(save_path = "plots/dwell_time.png")            # This opens the plot window
# plotter.plot_switching_times(save_path = "plots/switch_time.png")

# simulate trajectories 
print("simulating")
sim = TrajectorySimulator(k1star, k2, k3, epsilon, J, N,
                          max_steps=2, transient_steps=2)
sim.run_simulations()
print("done simulating")
sim.plot_results(save_path = "plots/trajectory.png")

results = sim.save_results()

params = {
    "k1star": sim.k1star,
    "k2": sim.k2,
    "k3": sim.k3,
    "epsilon": sim.epsilon,
    "J": sim.J,
    "N": sim.N,
}

saver = ResultsSaver()
saver.save(results, params)