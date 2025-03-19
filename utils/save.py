import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory import *

def save_pixelcnn(occupancy_map, dynamic_obstacles, gt_coefficients, pred_coefficients, filename="pixelcnn_visualization.png"):
    '''Save an image of the occupancy map with dynamic obstacles as circles,
       their velocities visualized as vectors, and the GT & predicted trajectories.''' 

    occupancy_map = occupancy_map.squeeze(0, 1).cpu().numpy()
    dynamic_obstacles = dynamic_obstacles.squeeze(0).cpu().numpy()
    gt_coefficients = gt_coefficients.squeeze(0).cpu().numpy()
    pred_coefficients = pred_coefficients.view(2, 11).squeeze(0).detach().cpu().numpy()
    # print(pred_coefficients)
    
    # Extract dynamic obstacles data
    x, y, u, v = dynamic_obstacles[0, 0, :], dynamic_obstacles[0, 1, :], dynamic_obstacles[0, 2, :], dynamic_obstacles[0, 3, :]
    
    # Find valid indices (assuming zero-padding)
    cut_idx = np.argmax(x == 0) if np.any(x == 0) else len(x)
    x, y, u, v = x[:cut_idx], y[:cut_idx], u[:cut_idx], v[:cut_idx]
    
    # Compute trajectories using Bernstein polynomials
    gt_trajectory_x, gt_trajectory_y = visualise_trajectory(gt_coefficients[0], gt_coefficients[1])
    pred_trajectory_x, pred_trajectory_y = visualise_trajectory(pred_coefficients[0], pred_coefficients[1])
    
    # Shift coordinates to center the occupancy map
    height, width = occupancy_map.shape
    height, width = height/10, width/10
    # x = x + width // 2
    # y = y + height // 2
    # gt_trajectory_x += width // 2
    # gt_trajectory_y += height // 2
    # pred_trajectory_x += width // 2
    # pred_trajectory_y += height // 2
    
    # Plot occupancy map
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(occupancy_map, cmap='gray', origin='lower', extent=[-width//2, width//2, -height//2, height//2])
    
    # Plot dynamic obstacles as circles
    for i in range(len(x)):
        circle = plt.Circle((x[i], y[i]), radius=0.1, color='r', fill=True, alpha=0.6)
        ax.add_patch(circle)
    
    # Plot velocity vectors
    ax.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=2, color='b')
    
    # Plot GT trajectory
    ax.plot(gt_trajectory_x, gt_trajectory_y, 'g-', label='GT Trajectory')
    
    # Plot predicted trajectory
    ax.plot(pred_trajectory_x, pred_trajectory_y, 'm--', label='Predicted Trajectory')
    
    ax.legend()
    ax.set_title("Occupancy Map with Dynamic Obstacles and Trajectories")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()