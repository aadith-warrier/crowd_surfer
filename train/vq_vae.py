import torch
from tqdm import tqdm
from termcolor import colored
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataloaders.vq_vae import VQVAEDataset
from models.vq_vae import VQVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_root = 'data/processed_sim_bags'

dataset = VQVAEDataset(root_dir='data/processed_sim_bags')
print(colored(f"Loaded VQ-VAE Dataset with {len(dataset)} samples", 'green'))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = VQVAE().to(device)

reconstruction_loss_func = torch.nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

writer = SummaryWriter()  # Initialize TensorBoard writer

beta=1e-5
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    loss_list = []
    reconstruction_loss_list = []
    quantizer_loss_list = []
    for batch in dataloader:
        coefficients, expert_trajectory = batch
        coefficients = coefficients.squeeze(0).permute(0, 2, 1).to(device)
        optimizer.zero_grad()

        pred_coefficients, quantizer_loss = model(coefficients)
        pred_coefficients = pred_coefficients.reshape(-1, 2, 11)
        reconstruction_loss = torch.sum(reconstruction_loss_func(pred_coefficients, coefficients))

        loss = quantizer_loss*beta + reconstruction_loss
        loss_list.append(loss.detach().cpu().item())
        reconstruction_loss_list.append(reconstruction_loss.detach().cpu().item())
        quantizer_loss_list.append(quantizer_loss.detach().cpu().item())

        loss.backward()
        optimizer.step()
    
    avg_loss = sum(loss_list) / len(loss_list)
    avg_recon = sum(reconstruction_loss_list) / len(reconstruction_loss_list)
    avg_quantizer = sum(quantizer_loss_list) / len(quantizer_loss_list)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Reconstruction Loss", avg_recon, epoch)
    writer.add_scalar("Quantizer Loss", avg_quantizer, epoch)

writer.close()  # Close the writer after training

torch.save(model, f='checkpoints/vqvae.pth')