import numpy as np
import matplotlib.pyplot as plt
from utils.trajectory import visualise_trajectory

# Load the coefficient data
file = 'data/processed_sim_bags/00000032/occupancy_map.npy'
data = np.load(file)

print(data.shape)