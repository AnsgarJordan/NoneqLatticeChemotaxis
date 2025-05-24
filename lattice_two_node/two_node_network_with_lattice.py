import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from IPython.display import HTML
import matplotlib.animation as animation


class Lattice_Two_Node:

    def __init__(self,
                 k_on,
                 k_off,
                 c,
                 alpha,
                 beta,
                 t_max,
                 key_int,
                 N
                 ):
        
        self.k_on = k_on
        self.k_off = k_off
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.t_max = t_max
        self.key = jax.random.PRNGKey(key_int)
        self.N = N

    def count_bound_neighbors(self, lattice):
        """Count number of bound neighbors for each site."""
        up = jnp.roll(lattice, shift=1, axis=0)
        down = jnp.roll(lattice, shift=-1, axis=0)
        left = jnp.roll(lattice, shift=1, axis=1)
        right = jnp.roll(lattice, shift=-1, axis=1)
        return up + down + left + right
    
    # ------------------------------
    # Gillespie-like KMC Step
    # ------------------------------
    def kmc_step(self, lattice, t, key):

        # grid of size = lattice that tells us how many neighbors are bound at that site 
        neighbors = self.count_bound_neighbors(lattice) 

        # add boolean mask for binding affinity
        is_unbound = (lattice == 0)
        is_bound = (lattice == 1)

        # k_on = rate, c = concentration, 1 + alpha * neighbors = cooperativity
        rate_on = self.k_on * self.c * (1 + self.alpha * neighbors) * is_unbound
        rate_off = self.k_off * (1 - self.beta * neighbors) * is_bound
        rate_off = jnp.clip(rate_off, a_min=0.0)  # avoid negative rates

        # Combine into a single rate matrix (how likely is each position to flip it's state?)
        rates = rate_on + rate_off
        total_rate = rates.sum()

        # If total_rate is zero, stop simulation (if no more simulations can happen, be done)
        def no_op():
            return lattice, t + 1e6, key

        def do_update():
            # Pick a random time increment
            key1, key2, key3 = jax.random.split(key, 3)
            r = jax.random.uniform(key1)
            delta_t = -jnp.log(r) / total_rate

            # Flatten rates and sample a site
            flat_rates = rates.ravel()
            idx = jax.random.choice(key2, flat_rates.size, p=flat_rates / total_rate)
            i, j = jnp.unravel_index(idx, lattice.shape)

            # Flip state
            new_state = 1 - lattice[i, j]
            new_lattice = lattice.at[i, j].set(new_state)
            return new_lattice, t + delta_t, key3

        return jax.lax.cond(total_rate > 0, do_update, no_op)

    def run_kmc_lattice(self, t_max, key):
        lattice = jnp.zeros((self.N, self.N), dtype=int)
        t = 0.0
        snapshots = [lattice]
        times = [t]

        # 1000 = 10 seconds of simulation
        for _ in range(500):  # max steps
            if (_ % 50 == 0):
                print(_)
            lattice, t, key = self.kmc_step(lattice, t, key)
            if t > t_max:
                break
            snapshots.append(lattice)
            times.append(t)

        return jnp.array(times), jnp.stack(snapshots)
    
    def make_animation(self, snapshots, times, ani_name = 'lattice_binding_simulation.mp4'):

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(snapshots[0], cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Lattice Binding State Over Time')

        text_overlay = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                            fontsize=12, color='black', 
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.axis('off')

        # Update function for animation
        def update(frame):
            lattice = snapshots[frame]
            # Compute and update bound fraction
            frac_bound = jnp.mean(lattice)
            text_overlay.set_text(f"Bound: {frac_bound * 100:.1f}%")
            im.set_data(snapshots[frame])
            ax.set_title(f"Time: {times[frame]:.2f} s")
            return [im]

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(times), interval=50, blit=True)

        ani.save(ani_name, writer='ffmpeg', fps=10)
