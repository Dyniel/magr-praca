import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import spectral_norm # Assuming spectral_norm is in utils

# Reusable building blocks, adapted from legacy/gan5.py
# These will now accept config parameters indirectly via their parent model's config.

class WSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spectral_norm=True):
        super().__init__()
        conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        if use_spectral_norm:
            self.conv = spectral_norm(conv_layer)
        else:
            self.conv = conv_layer

        # Weight scaling
        self.scale = math.sqrt(2 / (in_ch * kernel_size * kernel_size))
        nn.init.normal_(self.conv.weight, 0, 1) # Scaled by self.scale in forward
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x * self.scale)

class AdaIN(nn.Module):
    def __init__(self, style_dim, content_dim, use_spectral_norm=True):
        """
        Adaptive Instance Normalization.
        Args:
            style_dim (int): Dimension of the style input (e.g., from z).
                             In gan5, this was implicitly the content_dim for the MLP.
                             Here, we make it more general if needed, but for gan5's GCNBlock,
                             it's derived from content_dim.
            content_dim (int): Dimension of the content input (x).
        """
        super().__init__()
        # In gan5's GCNBlock, AdaIN's MLP input (x.mean) had 'dim' channels.
        # Let's assume dim here refers to content_dim.
        self.mlp = nn.Sequential(
            WSConv2d(content_dim, content_dim * 2, kernel_size=1, padding=0, use_spectral_norm=use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            WSConv2d(content_dim * 2, content_dim * 2, kernel_size=1, padding=0, use_spectral_norm=use_spectral_norm) # Output: gamma and beta
        )
        # No instance norm layer explicitly here, as it's part of the AdaIN definition:
        # gamma * ( (x - mu) / sigma ) + beta
        # However, gan5's AdaIN was: gamma * x + beta. This is more like FiLM layer or conditional batch norm without normalization.
        # For now, implementing as in gan5.
        # If true AdaIN is desired, an InstanceNorm2d would be needed before applying gamma and beta.

    def forward(self, content_features):
        # content_features: [B, C, S, 1] for GCNBlock
        style_params = self.mlp(content_features.mean([2,3], keepdim=True)) # [B, C*2, 1, 1]
        gamma, beta = style_params.chunk(2, 1) # Each [B, C, 1, 1]
        return gamma * content_features + beta


class GCNBlock(nn.Module):
    def __init__(self, dim, dropout_rate, use_ada_in, use_spectral_norm_conv=True):
        super().__init__()
        self.proj = WSConv2d(dim, dim, kernel_size=1, padding=0, use_spectral_norm=use_spectral_norm_conv)
        self.use_ada_in = use_ada_in
        if self.use_ada_in:
            self.ada = AdaIN(style_dim=dim, content_dim=dim, use_spectral_norm=use_spectral_norm_conv) # Matching gan5's implicit style/content dim
        else:
            self.ada = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, A):
        """
        Args:
            x (Tensor): Node features [B, C, S, 1] (S = num_superpixels)
            A (Tensor): Adjacency matrix [B, S, S]
        """
        B, C, S, _ = x.shape # S is num_superpixels

        h = self.proj(x) # [B, C, S, 1]

        # Graph convolution: A * H * W (here, H is features, W is weights of GCN layer)
        # Simplified: A * H (features)
        feat_for_gcn = h.squeeze(-1).permute(0, 2, 1) # [B, S, C]
        feat_after_gcn = torch.einsum('bij,bjk->bik', A, feat_for_gcn) # [B, S, C]

        h = feat_after_gcn.permute(0, 2, 1).unsqueeze(-1) # [B, C, S, 1]

        if self.ada:
            h = self.ada(h)

        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.dropout(h)
        return h


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_superpixels = config.num_superpixels # Store for easy access in forward

        # Input projection: (superpixel_color_features + z_dim) -> g_channels
        # Superpixel color features are 3 (RGB)
        self.proj_input = WSConv2d(3 + config.z_dim, config.g_channels, kernel_size=1, padding=0,
                                   use_spectral_norm=config.g_spectral_norm)

        self.gcn_blocks = nn.ModuleList()
        for _ in range(config.g_num_gcn_blocks): # e.g., 8 blocks
            self.gcn_blocks.append(
                GCNBlock(
                    dim=config.g_channels,
                    dropout_rate=config.g_dropout_rate,
                    use_ada_in=config.g_ada_in,
                    use_spectral_norm_conv=config.g_spectral_norm
                )
            )

        # Decoder / "Upsampler" (though it doesn't change spatial resolution in gan5)
        # Takes GCN output [B, g_channels, S, 1], scatters to [B, g_channels, H, W]
        # then processes to [B, 3, H, W]
        self.decoder = nn.Sequential(
            WSConv2d(config.g_channels, config.g_channels // 2, kernel_size=3, padding=1, use_spectral_norm=config.g_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            WSConv2d(config.g_channels // 2, 3, kernel_size=3, padding=1, use_spectral_norm=config.g_spectral_norm)
        )

        # Output normalization (InstanceNorm was used in gan5)
        if config.g_final_norm == 'instancenorm':
            self.final_norm = nn.InstanceNorm2d(3, affine=False) # Affine false is common for GANs
        elif config.g_final_norm == 'layernorm':
             # LayerNorm needs to know the shape. Assuming H, W are known.
             # This might be tricky if image_size can change dynamically without reinit.
             # For now, let's stick to InstanceNorm or None.
            self.final_norm = nn.InstanceNorm2d(3, affine=False) # Defaulting if layernorm chosen but not fully spec'd
            print("Warning: LayerNorm for Generator final_norm requested but not fully implemented, using InstanceNorm.")
        else: # 'none' or other
            self.final_norm = None


    def forward(self, z, real_images, segments_map):
        """
        Args:
            z (Tensor): Latent noise vector [B, z_dim].
            real_images (Tensor): Real images [B, 3, H, W], used to extract initial superpixel colors.
            segments_map (Tensor): Segmentation masks [B, H, W] (labels from 0 to S-1).
            A (Tensor): Adjacency matrices [B, S, S] - This was an input in gan5.
                        Now, A should be passed if GCNBlock needs it.
                        Let's assume A is available and passed. For now, I'll add A as an argument.
                        The alternative is that GCNBlock computes A if segments_map is given, but that's inefficient.
                        The dataset should provide A.
        """
        B, _, H, W = real_images.shape
        N = H * W # Total number of pixels
        S = self.num_superpixels # Number of superpixels, from config

        # 1. Aggregate initial superpixel features (mean color) from real_images
        flat_real_images = real_images.view(B, 3, N)         # [B, 3, N]
        flat_segments = segments_map.view(B, N)              # [B, N]

        # Create one-hot encoding for segments: [B, N, S]
        # Ensure segments_map has labels from 0 to S-1
        one_hot_segments = F.one_hot(flat_segments, num_classes=S).float()

        # Summing features per superpixel: one_hot.T @ flat_images.T
        # ( [B,S,N] @ [B,N,3] ) -> [B,S,3]
        sum_feats = torch.bmm(one_hot_segments.transpose(1, 2), flat_real_images.transpose(1, 2))

        # Counting pixels per superpixel
        counts = one_hot_segments.sum(dim=1).unsqueeze(-1)   # [B, S, 1]
        counts = counts.clamp(min=1e-6) # Avoid division by zero for empty superpixels

        mean_color_feats = sum_feats / counts                # [B, S, 3]

        # 2. Concatenate with latent vector z
        # Replicate z for each superpixel: [B, S, z_dim]
        z_replicated = z.view(B, 1, self.config.z_dim).expand(-1, S, -1)

        # Combined features: [B, S, 3 + z_dim]
        combined_superpixel_feats = torch.cat([mean_color_feats, z_replicated], dim=2)

        # Reshape for convolutional GCN blocks: [B, 3 + z_dim, S, 1]
        x = combined_superpixel_feats.permute(0, 2, 1).unsqueeze(-1)

        # 3. Initial projection
        x = self.proj_input(x) # [B, g_channels, S, 1]

        # 4. Pass through GCN blocks
        # This part needs the adjacency matrix A. It should come from the dataloader.
        # Let's modify forward to accept A.
        # The question is, where does A come from? The dataset loader.
        # So, forward should be: def forward(self, z, real_images, segments_map, adj_matrix)
        # For now, I'll assume adj_matrix is passed to forward.
        # If not, GCNBlock cannot work as intended.
        # The `gan5.py` GCNBlock took `A` as input.
        # The `Trainer` in `gan5.py` passed `adj` from batch to `G(z, real, seg, adj)`.
        # So, `adj_matrix` must be an input to this forward method.
        # I will add it now.
        # Error: `forward` signature changed, need to reflect this if called elsewhere.
        # No, `gan5.py`'s Generator forward was: `forward(self, z, images, segments, A)`
        # So `A` (adj_matrix) is already an argument. My previous comment was confused.
        # The current signature is `forward(self, z, real_images, segments_map, adj_matrix)`
        # which matches the intent.

        # Adjacency matrix A is expected as input now.
        # Let's assume the name is `adj_matrix` to match dataset output.
        # So, the call in trainer will be self.G(z, real_images, segments_map, adj_matrix_from_batch)

        # The GCNBlock in gan5 takes A.
        # `def forward(self, x, A):`
        # So when iterating self.gcn_blocks:
        # `x = block(x, adj_matrix_from_batch)`

        adj_matrix = None # Placeholder, this must be passed in.
                          # This is a major flaw in my current thought process for model definition standalone.
                          # The Generator's forward MUST take adj_matrix.
                          # Let me correct the signature of Generator.forward properly.

        # Corrected signature:
        # def forward(self, z, real_images, segments_map, adj_matrix):

        # Let's assume adj_matrix is passed as an argument.
        # The following line is incorrect, adj_matrix is an argument to this function.
        # adj_matrix = adj_matrix_from_batch # This would come from the trainer loop.

        for block in self.gcn_blocks:
            x = block(x, adj_matrix) # adj_matrix must be [B, S, S]

        # x is now [B, g_channels, S, 1]

        # 5. Scatter processed superpixel features back to pixel grid
        # [B, g_channels, S, 1] -> [B, S, g_channels]
        processed_superpixel_feats = x.squeeze(-1).permute(0, 2, 1)

        # Use one_hot_segments [B, N, S] to scatter: [B, N, S] @ [B, S, g_channels] -> [B, N, g_channels]
        pixel_level_feats = torch.bmm(one_hot_segments, processed_superpixel_feats)

        # Reshape to image-like tensor: [B, g_channels, H, W]
        pixel_level_feats = pixel_level_feats.permute(0, 2, 1).view(B, self.config.g_channels, H, W)

        # 6. Light decoder
        output_image = self.decoder(pixel_level_feats) # [B, 3, H, W]

        # 7. Final normalization
        if self.final_norm:
            output_image = self.final_norm(output_image)

        return torch.tanh(output_image) # Output in [-1, 1] range


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_channels = config.d_channels # Base number of channels for D

        layers = []
        # Initial conv: 3 -> d_channels
        # In gan5: Conv2d(3, d, 3, padding=1) - no stride, so image size maintained
        layers.append(WSConv2d(3, d_channels, kernel_size=3, stride=1, padding=1, use_spectral_norm=config.d_spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # Strided convolutions to reduce spatial dimensions
        # gan5: Conv2d(d, d * 2, 3, stride=2, padding=1) -> H/2, W/2
        # gan5: Conv2d(d * 2, d * 4, 3, stride=2, padding=1) -> H/4, W/4

        current_channels = d_channels
        # Example: if image_size=256, d_num_downsampling_layers = 2
        # 256 -> 128 (d_channels * 2)
        # 128 -> 64  (d_channels * 4)
        # This matches gan5 structure.

        # Let's make the number of downsampling layers configurable
        # For now, hardcoding to match gan5:

        # Layer 1 (d -> d*2, H/2)
        layers.append(WSConv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1, use_spectral_norm=config.d_spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_channels *= 2 # Now d_channels * 2

        # Layer 2 (d*2 -> d*4, H/4)
        layers.append(WSConv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1, use_spectral_norm=config.d_spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_channels *= 2 # Now d_channels * 4

        # Final layers from gan5: AdaptiveAvgPool2d(1), Flatten, Linear(d*4, 1)
        layers.append(nn.AdaptiveAvgPool2d(1)) # Output: [B, current_channels, 1, 1]
        layers.append(nn.Flatten()) # Output: [B, current_channels]

        # Final linear layer
        # In gan5, spectral_norm was applied to this Linear layer too.
        final_linear_layer = nn.Linear(current_channels, 1)
        if config.d_spectral_norm:
            layers.append(spectral_norm(final_linear_layer))
        else:
            layers.append(final_linear_layer)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # x is an image [B, 3, H, W]
        logits = self.main(x) # [B, 1]
        return logits.squeeze(1) # [B] for compatibility with BCEWithLogitsLoss if not using RaLSGAN

# Need to correct Generator forward signature to include adj_matrix
# Original gan5.py Generator: def forward(self, z, images, segments, A):
# My current Generator: def forward(self, z, real_images, segments_map):
# It's missing A (adj_matrix). I need to add it.

# Re-defining Generator with corrected forward signature
class Generator(nn.Module): # type: ignore
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_superpixels = config.num_superpixels

        self.proj_input = WSConv2d(3 + config.z_dim, config.g_channels, kernel_size=1, padding=0,
                                   use_spectral_norm=config.g_spectral_norm)

        self.gcn_blocks = nn.ModuleList()
        for _ in range(config.g_num_gcn_blocks):
            self.gcn_blocks.append(
                GCNBlock(
                    dim=config.g_channels,
                    dropout_rate=config.g_dropout_rate,
                    use_ada_in=config.g_ada_in,
                    use_spectral_norm_conv=config.g_spectral_norm
                )
            )

        self.decoder = nn.Sequential(
            WSConv2d(config.g_channels, config.g_channels // 2, kernel_size=3, padding=1, use_spectral_norm=config.g_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            WSConv2d(config.g_channels // 2, 3, kernel_size=3, padding=1, use_spectral_norm=config.g_spectral_norm)
        )

        if config.g_final_norm == 'instancenorm':
            self.final_norm = nn.InstanceNorm2d(3, affine=False)
        else:
            self.final_norm = None

    def forward(self, z, real_images, segments_map, adj_matrix): # Added adj_matrix
        B, _, H, W = real_images.shape
        N = H * W
        S = self.num_superpixels

        flat_real_images = real_images.view(B, 3, N)
        flat_segments = segments_map.view(B, N)
        one_hot_segments = F.one_hot(flat_segments, num_classes=S).float()

        sum_feats = torch.bmm(one_hot_segments.transpose(1, 2), flat_real_images.transpose(1, 2))
        counts = one_hot_segments.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
        mean_color_feats = sum_feats / counts

        z_replicated = z.view(B, 1, self.config.z_dim).expand(-1, S, -1)
        combined_superpixel_feats = torch.cat([mean_color_feats, z_replicated], dim=2)
        x = combined_superpixel_feats.permute(0, 2, 1).unsqueeze(-1)

        x = self.proj_input(x)

        for block in self.gcn_blocks:
            x = block(x, adj_matrix) # Pass adj_matrix to GCNBlock

        processed_superpixel_feats = x.squeeze(-1).permute(0, 2, 1)
        pixel_level_feats = torch.bmm(one_hot_segments, processed_superpixel_feats)
        pixel_level_feats = pixel_level_feats.permute(0, 2, 1).view(B, self.config.g_channels, H, W)

        output_image = self.decoder(pixel_level_feats)

        if self.final_norm:
            output_image = self.final_norm(output_image)

        return torch.tanh(output_image)

print("src/models.py created and populated with Generator, Discriminator, and helper modules.")
