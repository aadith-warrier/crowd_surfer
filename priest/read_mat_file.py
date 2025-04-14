from scipy.io import loadmat

file_path = '/home/aadith/Projects/Robotics/PRIEST/priest/src/obstacle_pos/obstacles_dy_21.mat'    

data = loadmat(file_path)

print(data.keys())

print(data['obs'][:, 0].shape)
