import torch
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from models.vq_vae import VQVAE
from models.fused import FusedModel

from dataloaders.pixel_cnn import PixelCNNDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = 'data/processed_sim_bags'

dataset = PixelCNNDataset(root_dir='data/processed_sim_bags')
print("Total Dataset Size:", len(dataset))
dataset = Subset(dataset, range(int(0.8*len(dataset))))
print("Taking Traing Subset:", len(dataset))
print(colored(f"Loaded PixelCNN Dataset with {len(dataset)} samples", 'green'))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

vqvae = torch.load('checkpoints/vqvae.pth').to(device)
vqvae.eval()
pixelcnn = FusedModel().to(device)

ce_loss_func = torch.nn.CrossEntropyLoss()

num_epochs = 500
optimizer = torch.optim.AdamW(pixelcnn.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-6)

writer = SummaryWriter(log_dir='./pixelcnn_512')  # Initialize TensorBoard writer

for epoch in range(num_epochs):
    loss_list = []
    for batch in tqdm(dataloader):
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
        # pixelcnn_logits = torch.nn.functional.softmax(pixelcnn_embedding, dim=2)

        with torch.no_grad():
            vqvae_embedding = vqvae.get_embedding(coeffecients)
            vqvae_idx = vqvae.get_indices(vqvae_embedding)
            # vqvae_one_hot = torch.nn.functional.one_hot(vqvae_idx, num_classes=16).float()

        loss = ce_loss_func(pixelcnn_embedding, vqvae_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.detach().cpu().item())
    
    avg_loss = sum(loss_list) / len(loss_list)
    print(f"Epoch {epoch} Loss: {avg_loss}")
    writer.add_scalar("Loss/train", avg_loss, epoch)     

torch.save(pixelcnn, f='checkpoints/pixelcnn_512.pth')
torch.save(pixelcnn.state_dict(), f='checkpoints/pixelcnn_512.pth')
