import torch.nn as nn
from src.models.blocks import EqualizedLinear

class SuperpixelLatentEncoder(nn.Module):
    def __init__(self, input_feature_dim, hidden_dims, output_embedding_dim, num_superpixels):
        super().__init__()
        self.num_superpixels = num_superpixels
        self.input_dim_total = input_feature_dim * num_superpixels

        layers = []
        current_dim = self.input_dim_total
        for h_dim in hidden_dims:
            layers.append(EqualizedLinear(current_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            current_dim = h_dim

        layers.append(EqualizedLinear(current_dim, output_embedding_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, mean_superpixel_features):
        B, S, F = mean_superpixel_features.shape
        if S != self.num_superpixels:
            raise ValueError(
                f"Input superpixels S={S} does not match encoder's expected num_superpixels={self.num_superpixels}")

        flat_features = mean_superpixel_features.view(B, -1)
        z_sp = self.mlp(flat_features)
        return z_sp
