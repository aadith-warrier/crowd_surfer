import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_plan(start, goal, obstacles, trajectory_x, trajectory_y, filename):

    static_x = obstacles.obstacle_trajectory_x[:, 0]
    static_y = obstacles.obstacle_trajectory_y[:, 0]

    dynamic_x = obstacles.dynamic_obstacle_trajectory_x[:, 0]
    dynamic_y = obstacles.dynamic_obstacle_trajectory_y[:, 0]

    #write code here to plot the static and dynamic_obstacles as ellipse of size 0.05m, 0.05m

        # Create the figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))

    ellipse_start = Ellipse((start.x, start.y), width=0.15, height=0.15, edgecolor='green', facecolor='green')
    ellipse_final = Ellipse((goal.x, goal.y), width=0.15, height=0.15, edgecolor='red', facecolor='red')

    ax.add_patch(ellipse_start)
    ax.add_patch(ellipse_final)

    # Plot static obstacles as ellipses (size 0.05m by 0.05m)
    for x, y in zip(static_x, static_y):
        ellipse = Ellipse((x, y), width=0.5, height=0.5, edgecolor='black', facecolor='black')
        ax.add_patch(ellipse)
    
    # Plot dynamic obstacles as ellipses (size 0.05m by 0.05m)
    for x, y in zip(dynamic_x, dynamic_y):
        ellipse = Ellipse((x, y), width=0.68, height=0.68, edgecolor='orange', facecolor='none', lw=2)
        ax.add_patch(ellipse)
    
    #plot the trajectory
    if trajectory_x.ndim == 1:
        plt.plot(trajectory_x, trajectory_y, 'r--')
    else:
        idx = np.random.choice(trajectory_x.shape[0], size=min(10,trajectory_x.shape[0]), replace=False)
        for i in idx:
            plt.plot(trajectory_x[i], trajectory_y[i], 'r')
    # Set plot limits and formatting
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.set_title("Static and Dynamic Obstacles")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.grid(True)

    # Save the plot as an image
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
