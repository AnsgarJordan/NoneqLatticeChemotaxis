##
# Two-Node network with alpha (increase this to increase the chance of binding) 
# and beta (increasing this decreases the chance of unbinding)
#


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from IPython.display import HTML
import matplotlib.animation as animation

from lattice_two_node_with_J_coupling.two_node_network_with_lattice_J import Lattice_Two_Node

def plot_multiple_fraction_bound(datasets, filename='fraction_bound_vs_time.png'):
    """
    Plot fraction of bound receptors over time for multiple (alpha, beta) values and save figure.
    
    Parameters:
    - datasets: list of tuples, each tuple contains:
        (snapshots, times, alpha, beta)
    - filename: name of the file to save the plot (default 'fraction_bound_vs_time.png')
    """

    plt.figure(figsize=(10, 5))

    for i, (snapshots, times, J) in enumerate(datasets):
        # Type check
        if not hasattr(snapshots, 'reshape'):
            raise TypeError(f"Dataset index {i} has invalid 'snapshots': expected array, got {type(snapshots)}")

        # Compute fraction bound for each timepoint
        frac_bound = jnp.mean(snapshots.reshape(len(times), -1), axis=1)
        label = f"J={J}"
        plt.plot(times, frac_bound, lw=2, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Fraction Bound")
    plt.title("Fraction of Bound Receptors Over Time")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(title="Coupling Parameters")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


## 
J_array = [0, 1, 2, 4]

t_max = 25
N = 20
k_on = 1
k_off = 0.1
c = 1
key_int = 0

datasets = []

# Loop through by index
for i in range(len(J_array)):
    J = J_array[i]
    key = jax.random.PRNGKey(key_int)

    model = Lattice_Two_Node(
        k_on=1.0,
        k_off=0.1,
        c=1.0,
        J=J,
        t_max=t_max,         # example short simulation
        key_int=key_int,
        N=20                # lattice size
    )

    times, snapshots = model.run_kmc_lattice(t_max=t_max, key=key)
    datasets.append((snapshots, times, J))

    model.make_animation(snapshots, times, ani_name = "animations/binding_animation_J=" + str(J)+ ".mp4")

plot_multiple_fraction_bound(datasets=datasets)