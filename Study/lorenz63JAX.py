"""
Lorenz63 Atmospheric Convection Model - JAX Implementation

This is a JAX-accelerated version of lorenz63.py that uses:
- Vectorized operations with jax.numpy for slightly faster performance
    - We will later see that it makes parallelizing computation much easier
- jax.lax.scan for faster trajectory generation

Key differences from the NumPy version:
- Uses jax.numpy instead of numpy (single precision floats, 32, by default)
- Implements a rollout() function for efficient batch trajectory generation
- Compatible with JAX transformations 


"""

import matplotlib.pyplot as plt
#import numpy as np
import jax
import jax.numpy as jnp


def f(u, *, sigma, rho, beta) -> jnp.ndarray:
    """
    Compute the derivative of the Lorenz system state vector.
    
    Args:
        u: State vector [x, y, z]
        sigma, rho, beta.
    
    Returns:
        Derivative vector [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = u
    x_dot = sigma * (y - x)
    y_dot = x * (rho - z) - y
    z_dot = x * y - beta * z
    return jnp.array([x_dot, y_dot, z_dot])

def rollout(stepper, n_iterations, *, include_init=False):
    """
    Create a function that generates trajectories using jax.lax.scan.
    
    This higher-order function transforms an autoregressive stepper function
    (one that takes state u_t and returns u_{t+1} like our lorenz stepper) into a batch trajectory generator that can be JIT-compiled for performance.
    
    Args:
        stepper: Callable that advances state by one step (u_t -> u_{t+1})
        n_iterations
        include_init: If True, add initial condition to trajectory
    
    Returns:
        rollout_function: Callable that takes u_0 and returns full trajectory
    """
    
    def scan_function(u, _):
        """
        Helper function for jax.lax.scan.
        
        Args:
            u: Current state vector
            _: Placeholder (scan requires this argument even if unused)
        
        Returns:
            (carry, output): Next state for both carry and stacking
        """
        u_next = stepper(u)
        return u_next, u_next
    
    def rollout_function(u_0):
        """
        Generate trajectory from initial condition/state.
        
        Args:
            u_0: Initial state vector
        
        Returns:
            trajectory: Array of shape (n_iterations, state_dim) 
        """
        _, trajectory = jax.lax.scan(
            scan_function,
            u_0,
            None,
            length = n_iterations
        )
        
        if include_init:
            return jnp.concatenate([jnp.expand_dims(u_0, axis = 0), trajectory])
        
        return trajectory
    
    return rollout_function
    

class LorenzSimulatorK4:
    """
    4th-order Runge-Kutta simulator for the Lorenz system.
    
    This class implements a single time-step integration using the RK4 method, helps to generate iterative trajectory.
    """
    
    def __init__(self, delta_T=0.01, *, sigma=10, rho=28, beta=8/3):
        """
        Initialize the Lorenz simulator.
        """
        self.dt = delta_T
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def __call__(self, u_previous) -> jnp.ndarray:
        """
        Advance the system by one time step using RK4.
        
        Args:
            u_previous: Current state vector u = [x, y, z]
        
        Returns:
            Next state vector [x, y, z] after time step dt
        """
        # Fix parameters for the Lorenz system
        f_fixed = lambda u: f(
            u,
            sigma=self.sigma,
            rho=self.rho,
            beta=self.beta
        )
        
        # RK4 intermediate steps
        k_1 = f_fixed(u_previous)
        k_2 = f_fixed(u_previous + 0.5 * self.dt * k_1)
        k_3 = f_fixed(u_previous + 0.5 * self.dt * k_2)
        k_4 = f_fixed(u_previous + self.dt * k_3)
        
        # Weighted combination for next state
        return u_previous + (self.dt / 6) * (k_1 + 2*k_2 + 2*k_3 + k_4)


def main():
    """
    Generate a trajectory of the Lorenz system.
    initial condition u_0 = [1, 1, 1].
    """
    # Initialize system parameters
    u_0 = jnp.ones(3)  # Initial state: [1, 1, 1]
    iterations = 5000
    lorenzStepper = LorenzSimulatorK4()
    
    # Generate trajectory using JAX rollout
    rollout_func = rollout(lorenzStepper, iterations, include_init=True)
    trajectory = rollout_func(u_0)
    
    # Create 3D visualization
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"},
        figsize=(7, 7)
    )
    
    # Plot the chaotic attractor trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        lw = 1.0,
        color = "blue",
        label = "Trajectory"
    )
    
    ax.scatter3D(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color = "red",
        label = "Starting point"
    )
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    ax.set_title("Lorenz Attractor (σ=10, ρ=28, β=8/3)")
    plt.legend()
    plt.show()


if __name__== "__main__":
    main()