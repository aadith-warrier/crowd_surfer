import torch
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataloaders.pixel_cnn import PixelCNNDataset

from models.vq_vae import VQVAE
from models.fused import FusedModel

from utils.save import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = 'data/processed_sim_bags'

dataset = PixelCNNDataset(root_dir='data/processed_sim_bags')
print("Total Dataset Size:", len(dataset))
dataset = Subset(dataset, range(int(0.8*len(dataset)), len(dataset)))
print("Taking Traing Subset:", len(dataset))
print(colored(f"Loaded PixelCNN Dataset with {len(dataset)} samples", 'green'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

vqvae = VQVAE()
vqvae = vqvae.to(device)
vqvae.load_state_dict(torch.load('checkpoints/state_dict/vqvae.pth'))
vqvae.eval()

pixelcnn = FusedModel()
pixelcnn = pixelcnn.to(device)
pixelcnn.load_state_dict(torch.load('checkpoints/state_dict/pixelcnn.pth'))
pixelcnn.eval()

for i, batch in enumerate(tqdm(dataloader)):
    #data io
    occupancy_grid, dynamic_obstacles, heading, coeffecients = batch
    occupancy_grid = occupancy_grid[:, 0, :, :].unsqueeze(1).float().to(device)
    dynamic_obstacles = dynamic_obstacles.permute(0, 1, 3, 2)[:, :, 1:, :].float().to(device)
    heading = heading[:, 0].unsqueeze(-1).float().to(device)
    coeffecients = coeffecients[:, 0, : ,:].permute(0, 2, 1).to(device)

    occupancy_grid[torch.isnan(occupancy_grid)] = 0
    dynamic_obstacles[torch.isnan(dynamic_obstacles)] = 0
    heading[torch.isnan(heading)] = 0

    #forward
    pixelcnn_embedding = pixelcnn(occupancy_grid, dynamic_obstacles, heading).permute(0, 2, 1)
    _, pixelcnn_idx = torch.max(pixelcnn_embedding, dim=1)
    # print("PixelCNN idx:", pixelcnn_idx)
    pred_traj = vqvae.from_indices(pixelcnn_idx)

    with torch.no_grad():
        vqvae_embedding = vqvae.get_embedding(coeffecients)
        vqvae_idx = vqvae.get_indices(vqvae_embedding)
        # print("VQVAE idx:", vqvae_idx)
    
    gt_traj = coeffecients
    save_pixelcnn(occupancy_grid, dynamic_obstacles, gt_traj, pred_traj, filename=f'outputs/pixelcnn/{i}.png')
    # exit()