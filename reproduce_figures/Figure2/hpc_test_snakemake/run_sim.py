# run_sim.py
import sys
import numpy as np
import ising_neq_v6_scan_functions_lax as ising
import param_config as param_config

deltaG_values = np.loadtxt("deltaG_values.txt")
index = int(sys.argv[1])
deltaG = deltaG_values[index]

# Define your constants
k1star = 1.0
k2 = 1.0
k3 = 1.0
J = 1.0
N = 6

flags = param_config.parser.parse_args("")
 
# Set parameters
N = flags.N
k2 = flags.k2
k3 = flags.k3
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
    **args_evaluating
)

results_dict = ising.convert_results_to_dict(results)

ising.save_results(results_dict, path = f"results/switch_{index}.npz")