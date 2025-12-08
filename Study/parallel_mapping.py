"""
Parallel Trajectory Generation for Lorenz63 System

This file demonstrates JAX's parallel mapping (vmap) to generate multiple Lorenz63
trajectories from different initial conditions simultaneously. We generate 9 trajectories starting from random initial states and visualize them in a 3x3 grid.

We learn:
    - jax.vmap: Vectorizes functions to operate on batches
    - Two equivalent approaches: vmap-then-rollout vs rollout-then-vmap
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from lorenz63JAX import LorenzSimulatorK4, rollout


def main():

    # 1: Initialize random initial conditions
    # Generate 9 random state vectors from standard normal distribution
    # Shape: (9, 3) where each row is an initial state [x, y, z]
    u_0_set = jax.random.normal(jax.random.PRNGKey(0), (9, 3))
    
    

    # 2: Generate trajectories using parallel mapping
    """
    the composition of vmap and rollout:
    
    Function signatures:
        - stepper:  (n,) -> (n,)
            Takes a nD state and returns the next nD state
        
        - rollout:  [(n,) -> (n,)] -> [(n,) -> (n_iterations, n)]
            Takes a stepper function and returns a function that generates
            a full trajectory of length n_iter from an initial state
        
        - vmap:  [(x,) -> (y,)] -> [(batch, x) -> (batch, y)]
            Vectorizes a function to operate on batches
    
    Two equivalent approaches:
        1. vmap-then-rollout: vmap(stepper) gives (9,3)->(9,3), then rollout
           Result shape: (n_iterations, 9, 3)
        
        2. rollout-then-vmap: rollout(stepper) gives (3,)->(n_iter,3), then vmap
           Result shape: (9, n_iter, 3)  [Standard convention: (batch, time, state)]
    """
    
    lorenzStepper = LorenzSimulatorK4()
    iterations = 5000
    
    # Approach 1: Apply vmap first, then rollout
    # Output shape: (time, batch, state) = (5001, 9, 3)
    trajectory_set_a = rollout(jax.vmap(lorenzStepper), iterations, include_init=True)(u_0_set)
    
    # Approach 2: Apply rollout first, then vmap (preferred)
    # Output shape: (batch, time, state) = (9, 5001, 3)
    rollout_func = rollout(lorenzStepper, iterations, include_init=True)
    trajectory_set_b = jax.vmap(rollout_func)(u_0_set)
    
    # Use approach 2 as it follows standard (batch, time, state) convention
    trajectory_set = trajectory_set_b
    

    # 3: Visualize all trajectories in a 3x3 grid

    fig, axs = plt.subplots(3, 3, subplot_kw = {"projection": "3d"}, figsize = (12,12))
    
    for i, ax in enumerate(axs.flat):
        # Plot the i-th trajectory
        ax.plot(
            trajectory_set[i,:,0],
            trajectory_set[i,:,1],
            trajectory_set[i,:,2],
        )
        ax.set_xlim(-20,20)
        ax.set_ylim(-30,30)
        ax.set_zlim(0,50)
        
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    plt.tight_layout()
    plt.show()


if __name__== "__main__":
    main()
