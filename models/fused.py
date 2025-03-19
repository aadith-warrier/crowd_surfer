import torch
from torchinfo import summary
import torch.nn as nn

from models.observation_embedding import ObservationEmbedding
from models.pixel_cnn import PixelCNN

class FusedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.observation_embedding = ObservationEmbedding(embedding_dim=16)
        self.pixel_cnn = PixelCNN(conditional_embedding_dim=16)

    def forward(self, occupancy_grid, dynamic_obstacles, heading):
        embedding = self.observation_embedding(occupancy_grid, dynamic_obstacles, heading)
        output = self.pixel_cnn(embedding.unsqueeze(1), embedding.unsqueeze(-1))
        return output

if __name__ == "__main__":
    model = FusedModel()
    summary(model, ((4, 1, 60, 60), (4, 5, 4, 10), (4, 1)))