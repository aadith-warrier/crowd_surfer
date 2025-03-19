import torch
from torchinfo import summary
from termcolor import colored

class Encoder(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 output_channels,
                 kernel_size,
                 num_hidden_layers,
                 embedding_size, 
                 embedding_dim):
        super().__init__()

        self.num_hidden_layers = num_hidden_layers

        self.input_layer = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, hidden_channels, kernel_size, padding=5),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU()
        )

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=5),
                                                   torch.nn.BatchNorm1d(hidden_channels),
                                                   torch.nn.ReLU()) 
                                                   for __ in range(num_hidden_layers)])
        
        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=5),
            torch.nn.BatchNorm1d(hidden_channels),
        )

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_embeddings,
                 beta):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def loss(self, x, quantized):
        commitment_loss = torch.nn.functional.mse_loss(quantized.detach(),x)
        codebook_loss = torch.nn.functional.mse_loss(x.detach(), quantized)
        loss = codebook_loss + self.beta*commitment_loss
        return loss

    def forward(self, x):
        indices = torch.argmin(torch.cdist(x.squeeze(-1), self.embedding.weight), dim=-1)
        quantized = self.embedding.forward(indices)
        return quantized, indices
    
    def from_indices(self, indices):
        quantized = self.embedding.forward(indices)
        return quantized
  
class Decoder(torch.nn.Module):
    def __init__(self, 
                 input_features,
                 hidden_features,
                 output_features,
                 num_hidden_layers):
        super().__init__()

        self.input_layer = torch.nn.Sequential(torch.nn.Linear(input_features, hidden_features), 
                                                torch.nn.BatchNorm1d(hidden_features),
                                                torch.nn.ReLU())

        self.hidden_layers = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(hidden_features, hidden_features), 
                                                    torch.nn.BatchNorm1d(hidden_features),
                                                    torch.nn.ReLU()) 
                                                    for __ in range(num_hidden_layers)])

        self.output_layer = torch.nn.Linear(hidden_features, output_features)

    def forward(self, x):
        out = self.input_layer(x)
        for layer in self.hidden_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out
    
class VQVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.kernel_size = 11
        self.hidden_channels = 96
        self.embedding_dim = 11
        self.embedding_size = 16
        self.output_channels = 96

        self.encoder = Encoder(input_channels=2,
                               hidden_channels=96,
                               output_channels=96,
                               kernel_size=11,
                               num_hidden_layers=8,
                               embedding_size=16,
                               embedding_dim=11)
        
        self.quantizer = VectorQuantizer(embedding_dim=self.embedding_dim,
                                         num_embeddings = self.embedding_size,
                                         beta=0.2)
        
        self.decoder = Decoder(input_features=96*11, 
                               hidden_features=1024,
                               output_features=22,
                               num_hidden_layers=8)

    def forward(self, x):

        embedding = self.encoder(x).squeeze(-1)

        quantized_embedding, __ = self.quantizer(embedding)

        quantized_embedding = embedding + (quantized_embedding - embedding).detach()
        quantizer_loss = self.quantizer.loss(embedding, quantized_embedding)

        out = self.decoder(quantized_embedding.reshape(quantized_embedding.shape[0], -1))

        return out, quantizer_loss
    
    def get_indices(self, embedding):
        _ , indices = self.quantizer(embedding)

        return indices

    def get_embedding(self, x):
        return self.encoder(x).squeeze(-1)

    def from_indices(self, indices):
        quantized_embedding = self.quantizer.from_indices(indices)
        out = self.decoder(quantized_embedding.reshape(quantized_embedding.shape[0], -1))
        return out

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    input_shape = [10, 2, 11]
    model = VQVAE().to(device)
    print(colored("Intialized VQ-VAE Model", 'green'))
    summary(model, input_shape)