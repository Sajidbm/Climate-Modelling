"""
Lorenz63 Atmospheric Convection Model

This file implements a numerical simulator for the Lorenz (1963) system of 
ordinary differential equations (ODEs), which models atmospheric convection and exhibits 
chaotic behavior.

System of equations:
    dx/dt = sigma*(y - x)
    dy/dt = x*(rho - z) - y
    dz/dt = xy - beta*z

Interpretation:
    - x: Rate of convection
    - y: Horizontal temperature variation
    - z: Vertical temperature variation

Lorenz's original values):
    - sigma = 10    : Prandtl number
    - rho  = 28    : Rayleigh number
    - beta  = 8/3   : Geometric factor

Method:
    The system is solved using the 4th-order Runge-Kutta (RK4) method:
    
    Given state vector u_t = [x_t, y_t, z_t] and f(u) = du/dt:
    
    k_1 = f(u_t)
    k_2 = f(u_t + Δt/2 · k_1)
    k_3 = f(u_t + Δt/2 · k_2)
    k_4 = f(u_t + Δt · k_3)
    
    u_{t+1} = u_t + Δt/6 · (k_1 + 2k_2 + 2k_3 + k_4)


"""

import matplotlib.pyplot as plt
import numpy as np


def f(u, *, sigma, rho, beta) -> np.ndarray:
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
    return np.array([x_dot, y_dot, z_dot])

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
    
    def __call__(self, u_previous) -> np.ndarray:
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
    # Initialize system at u_0 = [1, 1, 1]
    u_0 = np.ones(3)
    trajectory = [u_0]
    u_current = u_0
    
    # Generate trajectory using RK4 integration
    iterations = 5000
    lorenz = LorenzSimulatorK4()
    for i in range(iterations):
        u_current = lorenz(u_current)
        trajectory.append(u_current)
    trajectory = np.array(trajectory)
    
    # Create 3D visualization 
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"},
        figsize=(7, 7)
    )
    
    # Plot trajectory
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