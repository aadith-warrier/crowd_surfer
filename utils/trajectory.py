from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt

def bernstein_polynomial(v, n, x):
    """Compute the Bernstein basis polynomial B_v^n(x)"""
    return comb(n, v) * (x**v) * ((1 - x)**(n - v))

def visualise_trajectory(coefficients_x, coefficients_y):
    """
    Compute the X and Y coordinates of a trajectory using Bernstein polynomials.
    
    Arguments:
        coefficients_x: List of 11 coefficients for the X component.
        coefficients_y: List of 11 coefficients for the Y component.
    
    Returns:
        X, Y: Arrays representing the trajectory in 2D space.
    """
    x = np.linspace(0, 1, 100)  # Parameter range (0 to 1)
    X = np.zeros_like(x)
    Y = np.zeros_like(x)

    for i in range(11):
        X += coefficients_x[i] * bernstein_polynomial(i, 10, x)
        Y += coefficients_y[i] * bernstein_polynomial(i, 10, x)

    return X, Y

if __name__ == '__main__':
    # Generate random coefficients for a 2D trajectory
    coefficients_x = np.random.rand(11) * 10  # Scale to larger values
    coefficients_y = np.random.rand(11) * 10

    # Compute the trajectory
    X, Y = visualise_trajectory(coefficients_x, coefficients_y)

    # Plot the 2D trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(X, Y, label="Trajectory")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("2D Trajectory Visualization using Bernstein Polynomials")
    plt.legend()
    plt.grid()
    plt.show()