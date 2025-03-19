import torch

from tqdm import tqdm

from dataloaders.vq_vae import VQVAEDataset
from torch.utils.data import DataLoader
from models.vq_vae import VQVAE
import matplotlib.pyplot as plt

from utils.trajectory import visualise_trajectory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

save_dir = 'outputs/vqvae'

vqvae_dataset = VQVAEDataset(root_dir='data/processed_sim_bags')
dataloaders = DataLoader(vqvae_dataset, batch_size=1, shuffle=False, num_workers=1)

model = torch.load('checkpoints/vqvae.pth').to(device)
model.eval()

with torch.no_grad():
    counter = 0
    for i, batch in enumerate(dataloaders):
        coefficients, __ = batch
        coefficients = coefficients.squeeze(0).permute(0, 2, 1)
        print(f"Processing {i} of {len(dataloaders)}")
        for sample in tqdm(range(coefficients.shape[0])):
            coefficients = coefficients.to(device)
            pred_coefficients, __ = model(coefficients[sample].unsqueeze(0))
            pred_coefficients = pred_coefficients.view(-1, 2, 11).cpu().numpy()
            coefficients = coefficients.cpu()

            pred_coefficients_x = pred_coefficients[0, 0, :]
            pred_coefficients_y = pred_coefficients[0, 1, :]

            # Fix indexing issue: Extracting X and Y coefficients correctly
            coefficients_x = coefficients[sample][0][:].numpy()  # Corrected indexing
            coefficients_y = coefficients[sample][1][:].numpy()  # Corrected indexing

            # Compute the actual and predicted trajectories
            X, Y = visualise_trajectory(coefficients_x, coefficients_y)  # Ground Truth
            PX, PY = visualise_trajectory(pred_coefficients_x, pred_coefficients_y)  # Prediction

            # Plot both trajectories
            plt.figure(figsize=(8, 8))
            plt.plot(X, Y, label="Ground Truth Trajectory", linestyle="--", color="blue")
            plt.plot(PX, PY, label="Predicted Trajectory", linestyle="-", color="red")

            # Formatting the plot
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.title("2D Trajectories: Ground Truth vs. Prediction")
            plt.legend(fontsize=10)
            plt.grid()

            # Save image
            plt.savefig(f"outputs/vq_vae/{counter}.png", dpi=100)
            plt.close()
            counter+=1