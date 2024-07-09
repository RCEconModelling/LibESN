"""
Generate some basic datasets for testing.
"""

import numpy as np

def mackey_glass(beta, gamma, tau, n, initial_condition, dt, total_time):
    """
    Simulate the Mackey-Glass equations.

    Parameters:
        beta (float): Scaling factor.
        gamma (float): Rate of decay.
        tau (float): Time delay.
        n (float): Nonlinearity parameter.
        initial_condition (float): Initial condition for the system.
        dt (float): Time step for the simulation.
        total_time (float): Total time for the simulation.

    Returns:
        numpy.ndarray: Array containing the simulated data.
    """
    # Number of time steps
    num_steps = int(total_time / dt)

    # Initialize array to store results
    x = np.zeros(num_steps)
    x[0] = initial_condition

    # Simulation loop
    for t in range(1, num_steps):
        x_delayed = x[max(0, t - int(tau / dt))]
        x_dot = (beta * x_delayed) / (1 + x_delayed ** n) - gamma * x[t - 1]
        x[t] = x[t - 1] + x_dot * dt

    return x

def lorenz(sigma, rho, beta, initial_condition, dt, total_time):
    """
    Simulate the Lorenz attractor.

    Parameters:
        sigma (float): Parameter controlling the behavior of the system.
        rho (float): Parameter controlling the behavior of the system.
        beta (float): Parameter controlling the behavior of the system.
        initial_condition (numpy.ndarray): Initial condition for the system, shape (3,).
        dt (float): Time step for the simulation.
        total_time (float): Total time for the simulation.

    Returns:
        numpy.ndarray: Array containing the simulated data, shape (num_steps, 3).
    """
    # Number of time steps
    num_steps = int(total_time / dt)

    # Initialize array to store results
    data = np.zeros((num_steps, 3))
    data[0] = initial_condition

    # Simulation loop
    for t in range(1, num_steps):
        x, y, z = data[t - 1]

        dx_dt = sigma * (y - x)
        dy_dt = x * (rho - z) - y
        dz_dt = x * y - beta * z

        data[t] = data[t - 1] + np.array([dx_dt, dy_dt, dz_dt]) * dt

    return data

