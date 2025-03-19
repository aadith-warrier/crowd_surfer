import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory import visualise_trajectory

# Load the coefficient data
file = 'data/processed_sim_bags/00000032/best_priest_coefficients.npy'
data = np.load(file)

# Get the number of trajectories (limit to 50)
num_trajectories = min(data.shape[0], 50)

# Create the figure
plt.figure(figsize=(8, 8))

# Colormap for visualization
colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))

# Loop over multiple trajectories
for i in range(num_trajectories):
    coefficients_x = data[i, :, 0]
    coefficients_y = data[i, :, 1]

    # Compute trajectory
    X, Y = visualise_trajectory(coefficients_x, coefficients_y)

    # Plot with a unique color
    plt.plot(X, Y, color=colors[i], label=f"Trajectory {i+1}")

# Formatting the plot
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("2D Trajectories using Bernstein Polynomials")
plt.legend(fontsize=6, loc="best", ncol=2)  # Adjust legend for readability
plt.grid()
plt.show()