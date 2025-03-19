import torch
from torchinfo import summary
from torch import nn

from models.observation_embedding import ObservationEmbedding

class MaskedConv1D(torch.nn.Module):
    MASK_TYPES = ["A", "B"]

    def __init__(self, mask_type: str, conv1d: nn.Conv1d):
        super().__init__()
        assert mask_type in self.MASK_TYPES

        self.conv1d = conv1d

        self.register_buffer("mask", self.conv1d.weight.data.clone())
        self.mask: torch.Tensor
        __, __, size = self.conv1d.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, size//2 + (mask_type == 'B'):] = 0

    def forward(self, x):
        self.conv1d.weight.data *= self.mask
        output = self.conv1d.forward(x)
        return output

class ConditionedGatedMaskedConv1D(torch.nn.Module):
    def __init__(self, 
                 mask_type,
                 input_channel,
                 output_channel,
                 kernel,
                 padding,
                 conditional_embedding_dim,
                 bias=False):
        super().__init__()
        self.masked_conv1d_1 = MaskedConv1D(mask_type,
                                            nn.Conv1d(in_channels=input_channel,
                                                      out_channels=output_channel,
                                                      kernel_size=kernel,
                                                      padding=padding,
                                                      bias=bias))
        
        self.masked_conv1d_2 = MaskedConv1D(mask_type,
                                            nn.Conv1d(in_channels=input_channel,
                                                      out_channels=output_channel,
                                                      kernel_size=kernel,
                                                      padding=padding,
                                                      bias=bias))

        self.conditional_conv_1 = nn.Conv1d(in_channels=conditional_embedding_dim,
                                            out_channels=output_channel,
                                            kernel_size=kernel,
                                            padding=padding, 
                                            bias=bias)

        self.conditional_conv_2 = nn.Conv1d(in_channels=conditional_embedding_dim,
                                            out_channels=output_channel,
                                            kernel_size=kernel,
                                            padding=padding, 
                                            bias=bias)

    def forward(self, x, conditional_embedding):
        input = nn.functional.tanh(self.masked_conv1d_1(x) + self.conditional_conv_1(conditional_embedding))
        gate = nn.functional.sigmoid(self.masked_conv1d_2(x) + self.conditional_conv_2(conditional_embedding))
        return input*gate

class PixelCNN(torch.nn.Module):
    def __init__(self,
                 num_embeddings=96,
                 conditional_embedding_dim=16,
                 input_channels=1,
                 kernel_size=3,
                 padding=1,
                 hidden_channels=512,
                 num_layers=16,
                 residual_connection_freq=4):
        super().__init__()
        self.residual_connection_freq = residual_connection_freq
        self.input_layer = nn.ModuleList([ConditionedGatedMaskedConv1D(mask_type='A',
                                                                      input_channel=input_channels,
                                                                      output_channel=hidden_channels,
                                                                      kernel=kernel_size,
                                                                      padding=padding,
                                                                      conditional_embedding_dim=conditional_embedding_dim),
                                         nn.BatchNorm1d(hidden_channels)])
        
        self.layers = nn.ModuleList()
            
        for _ in range(num_layers):
            self.layers.extend([ConditionedGatedMaskedConv1D(mask_type='B',
                                                             input_channel=hidden_channels,
                                                             output_channel=hidden_channels,
                                                             kernel=kernel_size,
                                                             padding=padding,
                                                             conditional_embedding_dim=conditional_embedding_dim),
                                nn.BatchNorm1d(hidden_channels)])
        
        self.output_layer = nn.Conv1d(hidden_channels, num_embeddings, kernel_size=1)

    def forward(self, x, conditional_embedding):

        for layer in self.input_layer:
            if isinstance (layer, ConditionedGatedMaskedConv1D):
                x = layer(x, conditional_embedding)
            else:
                x = layer(x)

        residual = x

        for i, layer in enumerate(self.layers):
            if isinstance (layer, ConditionedGatedMaskedConv1D):
                x = layer(x, conditional_embedding)
            else:
                x = layer(x)

            if (i+1)%self.residual_connection_freq == 0:
                x += residual
                residual = x
                
        x = self.output_layer(x)
        return x
    
if __name__=='__main__':
    model = PixelCNN()
    summary(model, ((4, 1, 16), (4, 16, 1)))