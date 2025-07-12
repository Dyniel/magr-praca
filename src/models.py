import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import spectral_norm  # Assuming spectral_norm is in utils


# Reusable building blocks
# These will now accept config parameters indirectly via their parent model's config.

class WSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spectral_norm=True):
        super().__init__()
        conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        if use_spectral_norm:
            self.conv = spectral_norm(conv_layer)
        else:
            self.conv = conv_layer

        # Kaiming He initialization for LeakyReLU
        negative_slope = 0.2  # Assuming LeakyReLU with this slope is used after this conv
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        fan_in = in_ch * kernel_size * kernel_size
        std = gain / math.sqrt(fan_in)
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


# Note: AdaIN and GCNBlock were specific to gan5_gcn and are removed.
# If AdaIN is needed for other models like StyleGAN variants, it would be part of those model definitions.

# --- DCGAN Models ---

class DCGANGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model

        # Determine initial nz for the first ConvTranspose2d
        # config.model.z_dim is the dimension of the original noise vector
        # If latent conditioning is enabled, nz = z_dim + superpixel_latent_embedding_dim
        initial_nz = self.config_model.dcgan_z_dim  # Changed z_dim to dcgan_z_dim
        if self.config_model.dcgan_g_latent_cond:
            initial_nz += self.config_model.superpixel_latent_embedding_dim

        ngf = self.config_model.dcgan_g_feat  # Size of feature maps in generator
        nc = 3  # Number of channels in the output images (RGB)

        # Determine channels after first ConvTranspose2d and potential spatial concatenation
        # This is ngf*8 from G's own path, plus spatial_map_channels if C1 is active at this stage.
        channels_after_first_block = ngf * 8
        if self.config_model.dcgan_g_spatial_cond:  # C1: Spatial conditioning after first block
            channels_after_first_block += self.config_model.superpixel_spatial_map_channels_g
            # Note: The forward pass for C1 in DCGAN G is still marked as complex/skipped for this pass.
            # This __init__ change prepares for it if it were implemented by concat after first ConvT.

        self.main = nn.Sequential(
            # input is Z (potentially combined), going into a convolution
            nn.ConvTranspose2d(initial_nz, ngf * 8, 4, 1, 0, bias=False),  # nz is initial_nz
            # Output: [B, ngf*8, 4, 4]
            # If C1 (spatial_cond) active here, features are concatenated, channel dim becomes `channels_after_first_block`
            # The BatchNorm and subsequent ConvT must use this new channel count.
            nn.BatchNorm2d(channels_after_first_block),  # Adjusted for C1
            nn.ReLU(True),
            # state size. (channels_after_first_block) x 4 x 4
            nn.ConvTranspose2d(channels_after_first_block, ngf * 4, 4, 2, 1, bias=False),  # Adjusted for C1
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            # The output size of this layer depends on the target image_size.
            # For image_size 64, one more layer:
            # nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            # # state size. (nc) x 64 x 64

            # For image_size 256, we need more layers.
            # Current output is 32x32. Need to go to 256x256.
            # 32 -> 64 -> 128 -> 256 (3 more layers)
        )

        # Dynamically add layers based on image_size
        current_size = 32
        current_feat = ngf

        # For image_size 64:
        if config.image_size == 64:
            self.main.add_module("conv_transpose_out_64", nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
            self.main.add_module("tanh_out_64", nn.Tanh())
        elif config.image_size == 128:
            # From 32x32 (ngf)
            self.main.add_module("conv_transpose_64",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))  # 64x64
            self.main.add_module("bn_64", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_64", nn.ReLU(True))
            current_feat //= 2
            # From 64x64 (ngf/2)
            self.main.add_module("conv_transpose_out_128",
                                 nn.ConvTranspose2d(current_feat, nc, 4, 2, 1, bias=False))  # 128x128
            self.main.add_module("tanh_out_128", nn.Tanh())
        elif config.image_size == 256:
            # From 32x32 (ngf)
            self.main.add_module("conv_transpose_64",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))  # 64x64
            self.main.add_module("bn_64", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_64", nn.ReLU(True))
            current_feat //= 2
            current_size = 64
            # From 64x64 (ngf/2)
            self.main.add_module("conv_transpose_128",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))  # 128x128
            self.main.add_module("bn_128", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_128", nn.ReLU(True))
            current_feat //= 2
            current_size = 128
            # From 128x128 (ngf/4)
            self.main.add_module("conv_transpose_out_256",
                                 nn.ConvTranspose2d(current_feat, nc, 4, 2, 1, bias=False))  # 256x256
            self.main.add_module("tanh_out_256", nn.Tanh())
        else:
            # Default for original DCGAN paper (usually 64x64)
            # This part assumes image_size is 64 if not 128 or 256, adjust if other sizes are common
            if config.image_size != 32:  # if current size is not target
                raise ValueError(
                    f"DCGANGenerator: image_size {config.image_size} not directly supported by this dynamic structure. Base output is 32x32. Add more layers or check config.")
            else:  # if image_size is 32
                self.main.add_module("conv_transpose_out_32",
                                     nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False))  # Keep 32x32
                self.main.add_module("tanh_out_32", nn.Tanh())

    def forward(self, z_noise, graph_batch=None, real_images=None, segments_map=None, adj_matrix=None,
                spatial_map_g=None, z_superpixel_g=None):  # MODIFIED SIGNATURE
        # graph_batch, real_images, segments_map, adj_matrix are ignored by standard DCGAN
        # z_noise is expected to be of shape [batch_size, z_dim, 1, 1] for ConvTranspose2d

        # current_z_dim = self.config_model.dcgan_z_dim # This variable is not used, can be removed

        # C2: Latent Conditioning
        if self.config_model.dcgan_g_latent_cond and z_superpixel_g is not None:
            if z_noise.ndim == 4 and z_noise.shape[2] == 1 and z_noise.shape[3] == 1:
                z_noise = z_noise.squeeze(-1).squeeze(-1)
            elif z_noise.ndim != 2:
                raise ValueError(f"DCGAN G: z_noise has unexpected shape {z_noise.shape} for latent conditioning.")

            # Ensure z_superpixel_g is also 2D
            if z_superpixel_g.ndim != 2:
                raise ValueError(
                    f"DCGAN G: z_superpixel_g has unexpected shape {z_superpixel_g.shape} for latent conditioning, expected [B, embed_dim].")

            z_noise = torch.cat([z_noise, z_superpixel_g], dim=1)
            # The first ConvTranspose2d's in_channels (initial_nz) in __init__ must have been set to
            # dcgan_z_dim + superpixel_latent_embedding_dim for this to work.
            # This is handled by the __init__ logic.

        # Ensure z_noise is 4D before passing to the main network
        if z_noise.ndim == 2:
            z_noise = z_noise.unsqueeze(-1).unsqueeze(-1)  # Reshape to [B, combined_z_dim, 1, 1]

        # --- Handling of spatial_map_g (C1 conditioning) ---
        # The original code had a complex structure for C1.
        # It implied that self.main would be different if C1 was active.
        # For DCGAN, C1 (spatial conditioning) typically means concatenating the spatial map
        # to the feature maps at some layer. The __init__ logic for DCGANGenerator has been updated
        # to adjust channel sizes in self.main if dcgan_g_spatial_cond is true.
        # Specifically, it adjusts channels after the first ConvTranspose2d.
        # So, if C1 is active, self.main is already built to expect a concatenated feature map
        # after its first block.

        # The forward pass needs to implement this concatenation logic if C1 is active.
        # The current self.main is a nn.Sequential. We need to break it apart if C1 is active.

        if self.config_model.dcgan_g_spatial_cond and spatial_map_g is not None:
            # Pass z_noise through the first layer block (ConvTranspose2d, BatchNorm, ReLU)
            # self.main[0] is ConvTranspose2d(initial_nz, ngf * 8, 4, 1, 0, bias=False)
            # self.main[1] is BatchNorm2d(ngf * 8 or ngf * 8 + spatial_channels_g)
            # self.main[2] is ReLU(True)

            x = self.main[0](z_noise)  # Output: [B, ngf*8, 4, 4]

            # Concatenate spatial_map_g
            # spatial_map_g should be [B, C_sp_map, 4, 4]
            if x.shape[2:] != spatial_map_g.shape[2:]:
                # This check was present before, good to keep.
                # For DCGAN, if G's first feature map is 4x4, spatial_map_g should be 4x4.
                # The trainer code has logic to interpolate spatial_map_g to (4,4) if its channels > 0.
                raise ValueError(f"DCGAN G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match feature map shape {x.shape} for spatial conditioning at 4x4 stage.")

            x = torch.cat([x, spatial_map_g], dim=1)  # x is now [B, ngf*8 + C_sp_map, 4, 4]

            # Pass through the rest of the initially defined main network layers
            # self.main[1] (BatchNorm) and self.main[2] (ReLU) were already configured with the correct channel size in __init__
            x = self.main[1](x)  # BatchNorm
            x = self.main[2](x)  # ReLU

            # Now pass through the *rest* of the main sequence, starting from the third layer group
            # (the ConvTranspose2d that takes the (ngf*8 + C_sp_map) features)
            for i in range(3, len(self.main)):
                x = self.main[i](x)
            return x
        else:
            # If not using spatial conditioning, pass z_noise through the whole main network as is.
            return self.main(z_noise)


class DCGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        ndf = self.config_model.dcgan_d_feat  # Size of feature maps in discriminator

        initial_input_channels = 3
        if self.config_model.dcgan_d_spatial_cond:
            initial_input_channels += self.config_model.superpixel_spatial_map_channels_d

        # nc will be used as the input to the first conv layer in the dynamic loop below
        # For clarity, let's rename nc to current_in_c for the loop.
        # The first layer in the loop will use initial_input_channels.

        self.main = nn.Sequential(
            # input is (initial_input_channels) x image_size x image_size
            # For image_size 256:
            # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        )

        current_size = config.image_size
        current_feat_mult = 1  # For ndf, ndf*2, ndf*4 etc.

        # `in_c` starts with the potentially modified initial_input_channels
        in_c = initial_input_channels

        # Dynamically add layers based on image_size
        # Goal: reduce to 4x4 spatial dimension before final convolution
        while current_size > 4:
            # For the first conv layer, out_c is ndf. For subsequent, it's ndf * current_feat_mult (doubling).
            # The variable `current_feat_mult` is used to scale `ndf` for deeper layers.
            # Let's ensure the first layer outputs `ndf` channels, and subsequent layers scale up from there.

            out_c = ndf
            if current_feat_mult > 1:  # For layers after the first downsampling one that produces `ndf`
                out_c = ndf * (current_feat_mult // 2)  # current_feat_mult starts at 1, then becomes 2, 4, 8...
                # So this will be ndf, ndf*2, ndf*4...

            # Correction: The logic should be: first conv outputs ndf, next ndf*2, then ndf*4, etc.
            # So, out_c should use current_feat_mult directly for scaling ndf.
            # The loop structure implies `in_c` is what changes based on previous `out_c`.

            # Let's refine `out_c` determination:
            # Iteration 1: in_c = initial_input_channels, out_c = ndf
            # Iteration 2: in_c = ndf, out_c = ndf * 2
            # Iteration 3: in_c = ndf * 2, out_c = ndf * 4
            # ... up to ndf * 8

            if current_size == config.image_size:  # First convolutional layer
                out_c = ndf
            else:  # Subsequent layers
                # current_feat_mult will have been updated in previous iteration to reflect the *next* multiplier needed
                out_c = in_c * 2 if (in_c * 2) <= (ndf * 8) else (ndf * 8)

            self.main.add_module(f"conv_{current_size}x{current_size}",
                                 nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
            self.main.add_module(f"bn_{current_size}x{current_size}", nn.BatchNorm2d(out_c))
            self.main.add_module(f"lrelu_{current_size}x{current_size}", nn.LeakyReLU(0.2, inplace=True))

            in_c = out_c  # Next layer's input is current layer's output
            current_size //= 2

            # This part was for current_feat_mult scaling, which isn't directly used for out_c now.
            # if current_feat_mult == 1:
            #      current_feat_mult *=2
            # elif current_feat_mult < 8 :
            #      current_feat_mult *= 2

        # Final convolution
        # state size. (in_c) x 4 x 4, where in_c is the channel count of the 4x4 feature map
        self.main.add_module("conv_final_4x4", nn.Conv2d(in_c, 1, 4, 1, 0, bias=False))
        # No sigmoid here, as BCEWithLogitsLoss is used

    def forward(self, image, graph_data=None, spatial_map_d=None):
        # graph_data is ignored by standard DCGAN
        # image is the primary input [B, 3, H, W]
        # spatial_map_d is [B, C_sp_map, H, W]

        input_to_d = image
        if self.config_model.dcgan_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"DCGAN D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)

        return self.main(input_to_d).squeeze()  # Squeeze to remove extra dims for loss calculation


# --- StyleGAN2 Components ---

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.bias is not None:
            return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
        else:
            return F.linear(x, self.weight * self.lr_mul)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, n_layers=8, lr_mul=0.01):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=lr_mul))
            layers.append(nn.LeakyReLU(0.2))
        self.net = nn.Sequential(*layers)
        self.w_dim = w_dim

    def forward(self, z):
        # z shape: [batch_size, z_dim]
        return self.net(z)  # Output shape: [batch_size, w_dim]


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, demodulate=True, up=False, down=False,
                 blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.demodulate = demodulate
        self.down = down

        # Modulation: projects style vector to get per-channel scale for conv weights
        self.modulation = EqualizedLinear(style_dim, in_channels, bias=True)

        # Standard convolution weights (will be scaled by modulation)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        # Bias added after demodulation
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        # Upsampling/Downsampling related
        if down:
            factor = 2
            p = (factor - 1) + (kernel_size - 1)
            pad_val = p // 2
            self.blur = Blur(blur_kernel, pad=(pad_val, pad_val), downsample_factor=factor)

        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch_size, in_channels, height, width = x.shape

        # Modulate weights
        style = self.modulation(style).view(batch_size, 1, in_channels, 1, 1)  # [B, 1, C_in, 1, 1]
        weight = self.weight * style  # [B, C_out, C_in, K, K]

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)  # [B, C_out]
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)  # [B, C_out, C_in, K, K]

        weight = weight.view(  # Reshape for grouped convolution
            batch_size * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        if self.down:
            x = self.blur(x)
            x = x.view(1, batch_size * in_channels, x.shape[-2], x.shape[-1])
            out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
            out = out.view(batch_size, self.out_channels, out.shape[-2], out.shape[-1])
        else:
            # Standard convolution path
            # Input x: [B, C_in, H, W] -> [1, B*C_in, H, W] for grouped conv
            x = x.reshape(1, batch_size * in_channels, height, width)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
            # Output: [1, B*C_out, H', W'] -> [B, C_out, H', W']
            out = out.view(batch_size, self.out_channels, out.shape[-2], out.shape[-1])

        out = out + self.bias
        return out


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Scaling factor for the noise, learned per channel
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)  # Single channel noise broadcasted

        return x + self.weight * noise


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, downsample_factor=1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]  # Outer product to make 2D

        kernel /= kernel.sum()  # Normalize

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel[None, None, :, :])  # [1,1,K,K]
        self.pad = pad
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor

    def forward(self, x):
        # Pad
        # ReflectionPad2d might be better than ReplicationPad2d or ZeroPad2d
        # StyleGAN2 official uses custom padding. For simplicity, using F.pad with replicate.
        # The pad values might need to be (left, right, top, bottom)
        x = F.pad(x, (self.pad[0], self.pad[1], self.pad[0], self.pad[1]), mode='reflect')

        # Convolve
        # Assuming kernel is for single channel, expand to match input channels
        num_channels = x.size(1)
        kernel_expanded = self.kernel.expand(num_channels, -1, -1, -1)  # [C, 1, K, K]

        if self.upsample_factor > 1:
            # Transposed convolution for upsampling with blur
            # Note: This is a simplified blur for upsampling. Official StyleGAN2 uses a more complex FIR filter.
            x = F.conv_transpose2d(x, kernel_expanded, stride=self.upsample_factor, padding=0, groups=num_channels)
        elif self.downsample_factor > 1:
            x = F.conv2d(x, kernel_expanded, stride=self.downsample_factor, padding=0, groups=num_channels)
        else:  # Just blurring
            x = F.conv2d(x, kernel_expanded, padding=0, groups=num_channels)
        return x


class StyleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, kernel_size=3, upsample=False, blur_kernel=[1, 3, 3, 1],
                 num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.noises = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.upsample = upsample

        for i in range(num_layers):
            current_in = in_channel if i == 0 else out_channel
            self.convs.append(
                ModulatedConv2d(
                    current_in, out_channel, kernel_size, style_dim,
                    up=False, # Upsampling is now handled by F.interpolate
                    blur_kernel=blur_kernel
                )
            )
            self.noises.append(NoiseInjection(out_channel))
            self.activations.append(nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, style):
        # style is w: [B, style_dim]
        # noise_inputs is removed, noise is generated dynamically

        out = x
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        for i in range(self.num_layers):
            out = self.convs[i](out, style)
            out = self.noises[i](out, noise=None)  # Pass None to generate noise dynamically
            out = self.activations[i](out)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channels=3, kernel_size=1):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channels, kernel_size, style_dim, demodulate=False)
        # No activation after ToRGB in StyleGAN2, just conv + bias

    def forward(self, x, style, skip_rgb=None):
        # x: feature map from current block [B, C_in, H, W]
        # style: w vector [B, style_dim]
        # skip_rgb: upsampled RGB from previous block [B, 3, H, W]

        rgb = self.conv(x, style)
        if skip_rgb is not None:
            rgb = rgb + skip_rgb
        return rgb  # Typically not Tanh'ed here, Tanh at the very end of G.


class StyleGAN2Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size

        # For C2 (Latent Conditioning for G)
        actual_z_dim_for_mapping = self.config_model.stylegan2_z_dim
        if self.config_model.stylegan2_g_latent_cond:
            actual_z_dim_for_mapping += self.config_model.superpixel_latent_embedding_dim

        self.w_dim = self.config_model.stylegan2_w_dim
        self.n_mlp = self.config_model.stylegan2_n_mlp
        self.lr_mul_mapping = self.config_model.stylegan2_lr_mul_mapping

        self.channel_multiplier = self.config_model.stylegan2_channel_multiplier
        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
            512: 32 * self.channel_multiplier, 1024: 16 * self.channel_multiplier,
        }

        self.mapping_network = MappingNetwork(actual_z_dim_for_mapping, self.w_dim, n_layers=self.n_mlp,
                                              lr_mul=self.lr_mul_mapping)

        # Synthesis Network
        self.log_size = int(math.log2(self.image_size))
        # num_layers_total_for_w is used for broadcasting w if no style mixing, or for truncation.
        # It should be 2 styles per block (1 for StyleBlock, 1 for ToRGB) from 4x4 up to image_size.
        # Initial 4x4 block: 1 for initial_conv, 1 for initial_torgb.
        # Number of progressive blocks = log_size - 2 (e.g., for 256 (log_size=8), 6 blocks: 8x8 to 256x256)
        # Total styles = 2 (for 4x4) + (log_size - 2) * 2 = 2 * (log_size - 1)
        self.num_layers_total_for_w = 2 * (self.log_size - 1)

        # Initial block (e.g., 4x4)
        initial_block_channels = self.channels[4]
        if self.config_model.stylegan2_g_spatial_cond:  # C1: Spatial Conditioning for G
            initial_block_channels += self.config_model.superpixel_spatial_map_channels_g
            # The self.initial_constant would need to be handled carefully if its channels change.
            # Or, spatial_map_g is concatenated *after* initial_constant is repeated.

        # If C1, initial_constant's channels are self.channels[4].
        # The spatial_map_g is concatenated to it in forward before initial_conv.
        # So, initial_conv's in_channels must be self.channels[4] + C_sp_map_g
        self.initial_constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))

        self.initial_conv = StyleBlock(initial_block_channels, self.channels[4], self.w_dim, num_layers=1)
        self.initial_torgb = ToRGB(self.channels[4], self.w_dim)

        self.blocks = nn.ModuleList()
        self.torgbs = nn.ModuleList()

        in_ch = self.channels[4]
        for i in range(3, self.log_size + 1):  # From 8x8 (2^3) up to image_size (2^log_size)
            out_ch = self.channels[2 ** i]
            self.blocks.append(StyleBlock(in_ch, out_ch, self.w_dim, upsample=True))
            self.torgbs.append(ToRGB(out_ch, self.w_dim))
            in_ch = out_ch

        self.blur_kernel_default = [1, 3, 3, 1]  # Can be made configurable

        # Pre-generate noise structures if fixed noise is desired, or generate on the fly
        self.noises_fixed = None  # Placeholder for storing fixed noise if needed

    def w_to_styles(self, w, num_total_layers_for_w):
        # w: [B, w_dim] or [B, num_layers, w_dim] if already style-mixed
        if w.ndim == 2:  # if [B, w_dim]
            # Broadcast w to all layers: [B, num_total_layers_for_w, w_dim]
            return w.unsqueeze(1).repeat(1, num_total_layers_for_w, 1)
        elif w.ndim == 3 and w.shape[1] == num_total_layers_for_w:  # [B, num_layers, w_dim]
            return w
        else:
            raise ValueError(f"w has incompatible shape: {w.shape}")

    def forward(self, z_noise, style_mix_prob=0.9, input_is_w=False,
                truncation_psi=None, truncation_cutoff=None, w_avg=None,
                spatial_map_g=None, z_superpixel_g=None):  # New args for conditioning

        batch_size = z_noise.shape[0] if z_noise is not None else \
            (w_avg.shape[0] if input_is_w and w_avg is not None else  # Should determine batch_size robustly
             (spatial_map_g.shape[0] if spatial_map_g is not None else
              (z_superpixel_g.shape[0] if z_superpixel_g is not None else 1)))

        current_z = z_noise
        # C2: Latent Conditioning (applied to z before mapping network)
        if self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
            if current_z is None and input_is_w:  # Cannot easily combine if z_noise is not the primary input
                print(
                    "Warning (StyleGAN2 G): Latent superpixel conditioning (C2) requested but z_noise is None (input_is_w=True). C2 skipped.")
            elif current_z is not None:
                if current_z.ndim == 4 and current_z.shape[2:] == (1, 1): current_z = current_z.squeeze(-1).squeeze(-1)
                current_z = torch.cat([current_z, z_superpixel_g], dim=1)
            # else: z_superpixel_g is None, no C2 applied.

        if not input_is_w:
            if current_z is None:  # Should not happen if C2 is not applied when input_is_w
                raise ValueError("StyleGAN2 G: z_noise (current_z) is None and input_is_w is False.")
            w = self.mapping_network(current_z)  # [B, w_dim]
        else:  # input_is_w is True
            if current_z is not None and self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
                # This case is tricky: input_is_w=True means z_noise was actually w.
                # If C2 is also on, it implies current_z (which was w) was concatenated with z_sp.
                # This changes the meaning of 'w' passed to mapping_network.
                # This path needs careful thought if w is input AND C2 is on.
                # For now, assume if input_is_w, then current_z is already the final w or w_plus for style mixing.
                # And C2 should ideally be applied to the *original* z, not to an intermediate w.
                # Let's assume if input_is_w=True, C2 (z_superpixel_g) is ignored for safety,
                # unless the input 'z_noise' was specifically prepared as a pre-mapping latent.
                # The `actual_z_dim_for_mapping` in __init__ handles the mapping network's input size for C2.
                # So, if input_is_w=True, current_z should already be the correctly formed w.
                print(
                    "Warning/Info (StyleGAN2 G): input_is_w=True. Assuming z_noise arg is actually W. Latent superpixel conditioning (C2) might not behave as expected if z_noise was not the initial Z.")
                w = current_z  # current_z should be the W vector here.
            else:
                w = z_noise  # z_noise was directly the w vector [B, w_dim] or [B, num_layers, w_dim]

        if truncation_psi is not None and w_avg is not None:
            if w.ndim == 2:  # [B, w_dim]
                w_broadcast = w.unsqueeze(1).repeat(1, self.num_layers_total_for_w, 1)  # [B, num_layers, w_dim]
            else:  # [B, num_layers, w_dim]
                w_broadcast = w

            if truncation_cutoff is None:
                w_truncated = w_avg + truncation_psi * (w_broadcast - w_avg)
            else:
                w_前半 = w_avg + truncation_psi * (w_broadcast[:, :truncation_cutoff] - w_avg)
                w_後半 = w_broadcast[:, truncation_cutoff:]
                w_truncated = torch.cat([w_前半, w_後半], dim=1)
            styles = w_truncated  # [B, num_layers, w_dim]
        else:
            # Convert w to styles for each layer
            # Style mixing: with prob style_mix_prob, sample another z, map to w2, and use w2 for some layers
            if self.training and style_mix_prob > 0 and torch.rand(()).item() < style_mix_prob:
                if input_is_w:  # Cannot easily style mix if w is directly provided without z
                    print(
                        "Warning: StyleGAN2 G received input_is_w=True with style_mix_prob > 0. Style mixing skipped.")
                    styles = self.w_to_styles(w, self.num_layers_total_for_w)
                else:
                    z2 = torch.randn_like(z_noise)
                    if self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
                        z2 = torch.cat([z2, z_superpixel_g], dim=1)
                    w2 = self.mapping_network(z2)

                    # Choose a crossover point for style mixing
                    # num_total_layers_for_w counts style inputs for initial_conv (1) + each conv in blocks (num_blocks * layers_per_block)
                    mix_cutoff = torch.randint(1, self.num_layers_total_for_w, (1,)).item()

                    w_part1 = w.unsqueeze(1).repeat(1, mix_cutoff, 1)
                    w_part2 = w2.unsqueeze(1).repeat(1, self.num_layers_total_for_w - mix_cutoff, 1)
                    styles = torch.cat([w_part1, w_part2], dim=1)  # [B, num_layers, w_dim]
            else:  # No style mixing
                styles = self.w_to_styles(w, self.num_layers_total_for_w)  # [B, num_layers, w_dim]

        # --- Synthesis Network ---
        # Initial block (4x4)
        x = self.initial_constant.repeat(batch_size, 1, 1, 1)  # [B, C_const, 4, 4]

        # C1: Spatial Conditioning (concat to initial_constant features)
        if self.config_model.stylegan2_g_spatial_cond and spatial_map_g is not None:
            if x.shape[2:] != spatial_map_g.shape[2:]:
                # Resize spatial_map_g to match initial_constant's spatial dimensions (4x4)
                # This should ideally be done by the Trainer to ensure correct size.
                # For now, assume spatial_map_g is passed with H=4, W=4.
                # spatial_map_g = F.interpolate(spatial_map_g, size=x.shape[2:], mode='nearest')
                raise ValueError(f"StyleGAN2 G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match initial constant shape {x.shape} for spatial conditioning.")
            x = torch.cat([x, spatial_map_g], dim=1)  # x is now [B, C_const + C_sp_map, 4, 4]
            # self.initial_conv in __init__ was already set up with increased in_channels.

        # Style for initial_conv is styles[:, 0]
        x = self.initial_conv(x, styles[:, 0])
        rgb = self.initial_torgb(x, styles[:, 1])  # Style for initial_torgb is styles[:,1]

        # Style indexing: styles[:,0] for initial_conv, styles[:,1] for initial_torgb
        # styles[:,2] for first block in self.blocks, styles[:,3] for its ToRGB, etc.
        # styles[:, 2*k] for block k-1, styles[:, 2*k+1] for block k-1's ToRGB

        for i, (block, torgb) in enumerate(zip(self.blocks, self.torgbs)):
            style_for_block = styles[:, 2 + 2 * i]
            style_for_torgb = styles[:, 3 + 2 * i]

            x = block(x, style_for_block)

            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
            rgb = torgb(x, style_for_torgb, skip_rgb=rgb)

        return torch.tanh(rgb)  # Final output in [-1, 1]


class StyleGAN2Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size
        self.channel_multiplier = self.config_model.stylegan2_channel_multiplier
        # Channels match generator but often start smaller for D's input
        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
            512: 32 * self.channel_multiplier, 1024: 16 * self.channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))
        self.blur_kernel_default = [1, 3, 3, 1]  # Can be made configurable

        convs = []

        # FromRGB layer
        from_rgb_in_channels = 3
        if self.config_model.stylegan2_d_spatial_cond:  # C4: Spatial Conditioning for D
            from_rgb_in_channels += self.config_model.superpixel_spatial_map_channels_d

        convs.append(EqualizedConv2d(from_rgb_in_channels, self.channels[self.image_size], 1, activation='lrelu'))

        in_ch = self.channels[self.image_size]
        for i in range(self.log_size, 2, -1):  # Downsample from image_size to 4x4
            out_ch = self.channels[2 ** (i - 1)]  # Channels for the smaller resolution
            convs.append(ConvBlock(in_ch, out_ch, 3, downsample=True, blur_kernel=self.blur_kernel_default))
            in_ch = out_ch

        # Final 4x4 block
        convs.append(ConvBlock(in_ch, self.channels[4], 3))  # No downsample

        # Minibatch Standard Deviation (optional, but common in StyleGANs)
        # self.mbstd = MinibatchStdDev() # If implementing
        # convs.append(ConvBlock(self.channels[4] + (1 if self.mbstd else 0) , self.channels[4], 3))

        # Final prediction layer
        # Input to FC is features from 4x4 map.
        # If mbstd is used, channels[4]+1, else channels[4].
        # Let's assume no mbstd for now for simplicity.
        final_conv_channels = self.channels[4]
        convs.append(EqualizedConv2d(final_conv_channels, final_conv_channels, 4, padding=0))  # 4x4 to 1x1
        convs.append(nn.Flatten())
        convs.append(EqualizedLinear(final_conv_channels, 1))  # Output single logit

        self.convs = nn.Sequential(*convs)

    def forward(self, image, graph_data=None, spatial_map_d=None):  # New arg spatial_map_d
        # graph_data ignored for now

        input_to_d = image
        if self.config_model.stylegan2_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"StyleGAN2 D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)

        return self.convs(input_to_d).squeeze(1)


# Helper conv layers for Discriminator (simpler than ModulatedConv2d)
class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=None,
                 lr_mul=1.0):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.lr_mul = lr_mul

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) / lr_mul)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

        if activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        x = F.conv2d(x, self.weight * self.lr_mul, bias=(self.bias * self.lr_mul if self.bias is not None else None),
                     stride=self.stride, padding=self.padding)
        if self.activation:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):  # Used in Discriminator
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1],
                 activation='lrelu'):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channel, in_channel, kernel_size, padding=kernel_size // 2,
                                     activation=activation)

        if downsample:
            self.blur = Blur(blur_kernel, pad=((len(blur_kernel) - 1) // 2, (len(blur_kernel) - 1) // 2),
                             downsample_factor=2)  # Simplified padding
            # Conv after blur needs adjusted kernel size or padding if stride is used in conv
            # StyleGAN2 D uses strided conv for downsampling after blurring.
            # Here, blur handles downsampling, so conv stride is 1.
            self.conv2 = EqualizedConv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2,
                                         activation=activation)
        else:
            self.blur = None
            self.conv2 = EqualizedConv2d(in_channel, out_channel, kernel_size, padding=kernel_size // 2,
                                         activation=activation)

        self.downsample = downsample

    def forward(self, x):
        x = self.conv1(x)
        if self.downsample:
            x = self.blur(x)  # Blur might change padding requirements for conv2
        x = self.conv2(x)
        return x


# --- StyleGAN3 Components (Simplified) ---
# For a full StyleGAN3, dedicated FIR filtering and alias-free activations are crucial.
# The following are conceptual placeholders or simplified versions.

# CycleGAN Components have been removed.


# --- Projected GAN Components ---
import torchvision.models as tv_models
from torchvision.models.feature_extraction import create_feature_extractor


# It's good practice to ensure torchvision is available if used like this.
# A check or try-except could be added in a real scenario.

class FeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50", layers_to_extract=None, pretrained=True, requires_grad=False):
        super().__init__()
        if model_name == "resnet50":
            model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            if layers_to_extract is None:  # Default layers for ResNet50
                layers_to_extract = {
                    'relu': 'stem_relu',  # After initial conv and maxpool
                    'layer1': 'layer1_out',
                    'layer2': 'layer2_out',
                    'layer3': 'layer3_out',
                    'layer4': 'layer4_out',
                }
        elif model_name == "efficientnet_b0":
            model = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            if layers_to_extract is None:  # Example layers, inspect model.features for actual names
                layers_to_extract = {
                    'features.0': 'eff_feat0',  # Initial stem
                    'features.2': 'eff_feat1',  # End of MBConv block 2
                    'features.3': 'eff_feat2',  # End of MBConv block 3
                    'features.5': 'eff_feat3',  # End of MBConv block 5 (before last stage)
                    'features.8': 'eff_feat_final_conv',  # Output of final conv before pooling
                }
        else:
            raise ValueError(f"Unsupported feature_extractor model_name: {model_name}")

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers_to_extract)

        if not requires_grad:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()  # Set to eval mode if not training it

    def forward(self, x):
        # Input x should be normalized as expected by the pretrained model (e.g., ImageNet normalization).
        # Trainer must handle this.
        return self.feature_extractor(x)


class ProjectedGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size  # Needed for constructing D's own path if it's resolution dependent

        # --- Discriminator's own CNN path (e.g., similar to StyleGAN2's D) ---
        self.d_channel_multiplier = self.config_model.projectedgan_d_channel_multiplier
        self.d_channels = {  # Example, can be configured
            4: 512, 8: 512, 16: 256 * self.d_channel_multiplier, 32: 128 * self.d_channel_multiplier,
            64: 64 * self.d_channel_multiplier, 128: 32 * self.d_channel_multiplier,
            256: 16 * self.d_channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))

        d_convs = []
        # FromRGB layer for D's own path
        pgd_from_rgb_in_channels = 3
        if self.config_model.projectedgan_d_spatial_cond:  # C4 for ProjectedGAN D
            pgd_from_rgb_in_channels += self.config_model.superpixel_spatial_map_channels_d

        d_convs.append(
            EqualizedConv2d(pgd_from_rgb_in_channels, self.d_channels[self.image_size], 1, activation='lrelu'))

        in_ch = self.d_channels[self.image_size]
        for i in range(self.log_size, 2, -1):  # Downsample from image_size to 4x4
            out_ch = self.d_channels[2 ** (i - 1)]
            # Using ConvBlock from StyleGAN2's D components
            d_convs.append(
                ConvBlock(in_ch, out_ch, 3, downsample=True, blur_kernel=self.config_model.projectedgan_blur_kernel))
            in_ch = out_ch

        # Final 4x4 block for D's path
        d_convs.append(ConvBlock(in_ch, self.d_channels[4], 3))
        self.d_cnn_path = nn.Sequential(*d_convs)

        # Final linear layer for the adversarial logit from D's own features
        # This processes the 4x4 feature map from d_cnn_path
        self.final_d_conv = EqualizedConv2d(self.d_channels[4], self.d_channels[4], 4, padding=0,
                                            activation='lrelu')  # 4x4 to 1x1
        self.final_d_flatten = nn.Flatten()
        self.final_d_linear = EqualizedLinear(self.d_channels[4], 1)  # Single logit for adversarial loss

        # --- Pre-trained Feature Extractor (Frozen) ---
        # This is used for the feature matching loss for G, and D might also use its output.
        # For Projected GAN, D itself doesn't necessarily need to use these features for its adversarial decision,
        # but some variants might. The primary role of feature_net here is for the G's feature matching loss.
        # If D were to use these, it would need projection MLPs like in the original sketch.
        # For now, D only outputs its own adversarial logit. The features from feature_net
        # will be extracted and used by the Trainer for the G loss.

        # However, some Projected GAN versions *do* use the extracted features in D.
        # Let's assume a simple D that only outputs a logit, and the Trainer handles feature matching loss for G.
        # If D needs to use projected features, its architecture gets more complex with fusion.

    def forward(self, image_for_d_path):
        # image_for_d_path is for D's own processing, typically in [-1, 1]

        # D's own CNN path
        d_features = self.d_cnn_path(image_for_d_path)
        h = self.final_d_conv(d_features)
        h = self.final_d_flatten(h)
        logit = self.final_d_linear(h)

        return logit.squeeze(1)
        # If D were to also return its own features for some reason:
        # return logit.squeeze(1), d_features (or features from other D layers)

    # The forward method of ProjectedGANDiscriminator needs to be updated for C4
    # It should accept spatial_map_d and concatenate it if the flag is set.
    # Let's redefine its forward method here.
    def forward(self, image_for_d_path, spatial_map_d=None):  # Added spatial_map_d
        input_to_d_cnn = image_for_d_path
        if self.config_model.projectedgan_d_spatial_cond and spatial_map_d is not None:
            if image_for_d_path.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"ProjectedGAN D: image shape {image_for_d_path.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d_cnn = torch.cat([image_for_d_path, spatial_map_d], dim=1)

        d_features = self.d_cnn_path(input_to_d_cnn)
        h = self.final_d_conv(d_features)
        h = self.final_d_flatten(h)
        logit = self.final_d_linear(h)

        return logit.squeeze(1)


class ProjectedGANGenerator(StyleGAN2Generator):
    def __init__(self, config):
        # Generator for Projected GAN is often a StyleGAN2 generator.
        # We pass the main config, and StyleGAN2Generator constructor uses config.model.stylegan2_* params.
        # We need to ensure that the ProjectedGAN config section in base_config.py
        # also defines these stylegan2_* parameters, or we create aliases.
        # For simplicity, let's assume the config structure will provide them under config.model.
        # e.g. config.model.stylegan2_z_dim will be used.
        super().__init__(config)  # This will use config.model.stylegan2_... parameters.
        # This means the YAML for ProjectedGAN should define these.
        print("Initialized ProjectedGANGenerator (based on StyleGAN2Generator).")

    # The forward pass is inherited from StyleGAN2Generator.
    # The key difference for Projected GAN's generator is how it's trained (via feature matching loss).


# Note on Normalization for FeatureExtractor:
# The Trainer will be responsible for normalizing images to ImageNet stats
# *before* passing them to the FeatureExtractor (which will be a separate module instance in the Trainer).
# The ProjectedGANDiscriminator defined above takes images in the GAN's native range (e.g. [-1,1])
# for its own CNN path.

# --- Superpixel Latent Encoder (for C2 Conditioning) ---
class SuperpixelLatentEncoder(nn.Module):
    def __init__(self, input_feature_dim, hidden_dims, output_embedding_dim, num_superpixels):
        super().__init__()
        self.num_superpixels = num_superpixels
        self.input_dim_total = input_feature_dim * num_superpixels  # Flattened features from all superpixels

        layers = []
        current_dim = self.input_dim_total
        for h_dim in hidden_dims:
            layers.append(EqualizedLinear(current_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            current_dim = h_dim

        layers.append(EqualizedLinear(current_dim, output_embedding_dim))
        # No activation on the final output embedding generally

        self.mlp = nn.Sequential(*layers)

    def forward(self, mean_superpixel_features):
        """
        Args:
            mean_superpixel_features (torch.Tensor): Batch of mean features per superpixel.
                                                     Shape: [B, S, FeatureDim]
                                                     (e.g., B, num_superpixels, 3 for RGB)
        Returns:
            torch.Tensor: Superpixel latent embedding z_sp. Shape: [B, output_embedding_dim]
        """
        B, S, F = mean_superpixel_features.shape
        if S != self.num_superpixels:
            # This could happen if the actual number of superpixels in an image varies.
            # For a fixed MLP input size, we might need to pad/truncate or use a different encoder type (e.g., DeepSets, GNN).
            # For now, assume S is fixed and matches self.num_superpixels.
            raise ValueError(
                f"Input superpixels S={S} does not match encoder's expected num_superpixels={self.num_superpixels}")

        flat_features = mean_superpixel_features.view(B, -1)  # [B, S * FeatureDim]
        z_sp = self.mlp(flat_features)
        return z_sp


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0, learnable=False):
        super().__init__()
        # Mapping from w (e.g., 512) to a higher dimensional space for Fourier features
        # These are then used as the input to the first synthesis layer.
        # This replaces the learned constant input of StyleGAN2.
        self.linear = EqualizedLinear(in_features, out_features, lr_mul=1.0)  # W to Fourier feature basis
        # Actual Fourier feature generation (sin/cos) happens in the forward pass based on w.
        # StyleGAN3 uses a specific grid of frequencies.
        # This is a simplified placeholder for the concept.
        # A more complete implementation would involve defining the frequency bands and phases.
        self.out_features = out_features

    def forward(self, w):
        # w: [B, in_features] (typically w from mapping network)
        # This is highly simplified. StyleGAN3 constructs a grid of coordinates
        # and uses w to modulate phases/amplitudes of sinusoidal basis functions on that grid.
        # Here, we'll just pass w through a linear layer and then use that as input to the first conv.
        # The "Fourier" aspect is more about the conceptual input representation.
        fourier_basis = self.linear(w)  # [B, out_features]
        # This fourier_basis would then be shaped into a spatial tensor, e.g., [B, C, H_init, W_init]
        # For StyleGAN3, H_init, W_init are typically small (e.g., 4x4 or derived from w directly)
        # This part needs significant expansion for a real StyleGAN3.
        # For now, let's assume this directly produces the input features for the first layer.
        # The actual spatial generation from w is more complex.
        # Let's assume out_features is already C*H*W for the first layer.
        # For a 4x4 start with C channels: out_features = C * 4 * 4
        return fourier_basis


class AliasFreeActivation(nn.Module):
    # Placeholder for an alias-free activation like filtered LeakyReLU
    def __init__(self, negative_slope=0.2, upsample_factor=2, downsample_factor=1, fir_kernel=None):
        super().__init__()
        self.negative_slope = negative_slope
        # In a real implementation, this would involve FIR filtering before/after activation
        # or a custom activation function designed to be alias-free.
        # For simplicity, using standard LeakyReLU.
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        # FIR filtering part (conceptual)
        self.use_fir = fir_kernel is not None
        if self.use_fir:
            # Simplified blur as a stand-in for proper FIR filtering
            # This would be more complex (e.g., using upfirdn2d from StyleGAN2-ADA or StyleGAN3 official)
            pad_size = (len(fir_kernel) - 1) // 2
            self.fir_filter_up = Blur(fir_kernel, pad=(pad_size, pad_size),
                                      upsample_factor=upsample_factor) if upsample_factor > 1 else None
            self.fir_filter_down = Blur(fir_kernel, pad=(pad_size, pad_size),
                                        downsample_factor=downsample_factor) if downsample_factor > 1 else None

    def forward(self, x):
        # Conceptual flow: filter -> activate -> filter
        # Simplified: just activate
        # if self.fir_filter_down: x = self.fir_filter_down(x) # Filter before non-linearity if downsampling
        x = self.activation(x)
        # if self.fir_filter_up: x = self.fir_filter_up(x) # Filter after non-linearity if upsampling
        return x


class StyleGAN3Layer(nn.Module):
    # A single layer in StyleGAN3's synthesis network (highly simplified)
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3, upsample=False, fir_kernel=None):
        super().__init__()
        self.upsample = upsample
        # Modulation and Conv (similar to StyleGAN2's ModulatedConv2d but without some complexities initially)
        self.style_affine = EqualizedLinear(w_dim, in_channels, bias=True)  # Affine transform for style

        # Convolution: In StyleGAN3, this is often an equivariant conv or carefully designed standard conv
        # For simplicity, using EqualizedConv2d from StyleGAN2 components
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size,
                                    padding=kernel_size // 1)  # padding kernel_size//2

        self.noise_injection = NoiseInjection(out_channels)  # Similar to StyleGAN2
        self.activation = AliasFreeActivation(negative_slope=0.2, upsample_factor=(2 if upsample else 1),
                                              fir_kernel=fir_kernel)

        if upsample:
            # Simplified upsampling, StyleGAN3 uses specific FIR-based upsamplers
            # This would ideally be integrated with the convolution (strided transpose conv with FIR)
            self.upsampler_fir = Blur(fir_kernel if fir_kernel else [1, 3, 3, 1],
                                      pad=((len(fir_kernel) - 1) // 2 if fir_kernel else 1,
                                           (len(fir_kernel) - 1) // 2 if fir_kernel else 1), upsample_factor=2)

    def forward(self, x, w_style, noise=None):
        # Style modulation (simplified)
        style = self.style_affine(w_style).unsqueeze(2).unsqueeze(3)  # [B, C_in, 1, 1]
        x = x * (style + 1)  # Affine: scale and shift (bias is in style_affine)

        if self.upsample:
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # Basic upsample
            x = self.upsampler_fir(x)  # Upsample with FIR blur

        x = self.conv(x)
        # Always generate noise dynamically after potential upsampling.
        x = self.noise_injection(x, noise=None)
        x = self.activation(x)
        return x


class StyleGAN3Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size

        actual_z_dim_for_mapping = self.config_model.stylegan3_z_dim
        if self.config_model.stylegan3_g_latent_cond:  # C2 for StyleGAN3 G
            actual_z_dim_for_mapping += self.config_model.superpixel_latent_embedding_dim

        self.w_dim = self.config_model.stylegan3_w_dim
        self.n_mlp = self.config_model.stylegan3_n_mlp
        self.lr_mul_mapping = self.config_model.stylegan3_lr_mul_mapping
        self.channel_multiplier = self.config_model.stylegan3_channel_multiplier
        self.fir_kernel = self.config_model.stylegan3_fir_kernel

        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
        }

        self.mapping_network = MappingNetwork(actual_z_dim_for_mapping, self.w_dim, n_layers=self.n_mlp,
                                              lr_mul=self.lr_mul_mapping)

        self.log_size = int(math.log2(self.image_size))
        self.num_styles_needed = 2 * (
                    self.log_size - 1)  # Similar to StyleGAN2's w broadcasting logic for styles per layer

        # --- Synthesis Network (StyleGAN3-T like structure) ---
        self.fourier_features_dim = self.channels[4] * 4 * 4
        self.fourier_input = FourierFeatures(self.w_dim,
                                             self.fourier_features_dim)  # w_dim -> features for 4x4 C channels

        # Determine input channels for the first StyleGAN3Layer
        self.initial_reshape_channels = self.channels[4]  # Channels from Fourier features after reshaping
        first_layer_in_channels = self.initial_reshape_channels
        if self.config_model.stylegan3_g_spatial_cond:  # C1 for StyleGAN3 G
            first_layer_in_channels += self.config_model.superpixel_spatial_map_channels_g

        self.layers = nn.ModuleList()
        self.torgbs = nn.ModuleList()  # ToRGB layers, StyleGAN3 often has one at the end or per "magnitude" group

        in_ch = self.initial_reshape_channels
        current_res = 4

        # Initial layer (no upsampling)
        # Uses first_layer_in_channels which might include spatial conditioning channels
        self.layers.append(
            StyleGAN3Layer(first_layer_in_channels, self.channels[current_res], self.w_dim, fir_kernel=self.fir_kernel))
        in_ch = self.channels[current_res]  # Output of first layer becomes input for next

        # Progressive layers
        for i in range(3, self.log_size + 1):  # from 8x8 up to image_size
            target_res = 2 ** i
            out_ch = self.channels[target_res]
            self.layers.append(StyleGAN3Layer(in_ch, out_ch, self.w_dim, upsample=True, fir_kernel=self.fir_kernel))
            # self.torgbs.append(ToRGB(out_ch, self.w_dim)) # One ToRGB per resolution
            in_ch = out_ch

        # Final ToRGB layer (StyleGAN3 often has a single final ToRGB or more complex output stage)
        self.final_torgb = ToRGB(in_ch, self.w_dim)  # Using StyleGAN2's ToRGB for simplicity

    def make_noise_sg3(self, batch_size):
        noises = []
        current_res = 4  # Initial resolution
        # Noise for initial layer
        noises.append(
            torch.randn(batch_size, 1, current_res, current_res, device=self.mapping_network.net[1].weight.device))
        for i in range(3, self.log_size + 1):
            current_res *= 2
            noises.append(
                torch.randn(batch_size, 1, current_res, current_res, device=self.mapping_network.net[1].weight.device))
        return noises

    def forward(self, z_noise, noise_inputs=None, input_is_w=False, truncation_psi=None, w_avg=None,
                spatial_map_g=None, z_superpixel_g=None):  # New conditioning args

        # Determine batch_size (similar to StyleGAN2Generator)
        if z_noise is not None:
            batch_size = z_noise.shape[0]
        elif input_is_w and w_avg is not None:
            batch_size = w_avg.shape[0]  # Should be z_noise if input_is_w means z_noise is w
        elif input_is_w and z_noise is None:  # z_noise here is actually w
            if isinstance(spatial_map_g, torch.Tensor):
                batch_size = spatial_map_g.shape[0]  # Try to infer from other inputs
            elif isinstance(z_superpixel_g, torch.Tensor):
                batch_size = z_superpixel_g.shape[0]
            else:
                batch_size = 1  # Fallback, might be incorrect
        elif isinstance(spatial_map_g, torch.Tensor):
            batch_size = spatial_map_g.shape[0]
        elif isinstance(z_superpixel_g, torch.Tensor):
            batch_size = z_superpixel_g.shape[0]
        else:
            batch_size = 1  # Default if cannot infer

        current_z = z_noise
        # C2: Latent Conditioning (applied to z before mapping network)
        if self.config_model.stylegan3_g_latent_cond and z_superpixel_g is not None:
            if current_z is None and input_is_w:
                print(
                    "Warning (StyleGAN3 G): Latent superpixel conditioning (C2) requested but z_noise is None (input_is_w=True). C2 skipped.")
            elif current_z is not None:
                if current_z.ndim == 4 and current_z.shape[2:] == (1, 1): current_z = current_z.squeeze(-1).squeeze(-1)
                current_z = torch.cat([current_z, z_superpixel_g], dim=1)

        if not input_is_w:
            if current_z is None:
                raise ValueError("StyleGAN3 G: z_noise (current_z) is None and input_is_w is False.")
            w = self.mapping_network(current_z)
        else:  # input_is_w is True
            # Similar logic to StyleGAN2 G for handling w when input_is_w=True and C2 is active
            if current_z is not None and self.config_model.stylegan3_g_latent_cond and z_superpixel_g is not None:
                print(
                    "Warning/Info (StyleGAN3 G): input_is_w=True. Assuming z_noise arg is W. C2 might not behave as expected.")
                w = current_z
            else:
                w = z_noise  # z_noise was directly the w vector

        # Truncation (simplified, assumes w is [B, w_dim] or [B, 1, w_dim] for broadcasting)
        # StyleGAN3's truncation might be different due to Fourier features.
        # For now, use same logic as StyleGAN2 if w_avg is provided.
        # This requires w to be broadcastable to all layers that need it.
        # Let's assume one w is used for all layers for now (no style mixing across layers here)

        current_w = w
        if w.ndim == 3 and w.shape[1] == 1:  # [B, 1, w_dim]
            current_w = w.squeeze(1)  # [B, w_dim]
        elif w.ndim == 3 and w.shape[1] > 1:
            # If multiple w vectors are provided (e.g. for style mixing per layer, not implemented yet in G3 forward)
            # We'd need to select the appropriate w for each layer. For now, use the first one.
            print("Warning: StyleGAN3 G received multiple w vectors per sample, using w[:,0] for all layers.")
            current_w = w[:, 0, :]

        if truncation_psi is not None and w_avg is not None:
            current_w = w_avg + truncation_psi * (current_w - w_avg)

        # --- Synthesis ---
        # Initial Fourier features -> spatial tensor
        x = self.fourier_input(current_w)  # [B, C_fourier_flat]
        x = x.view(batch_size, self.initial_reshape_channels, 4, 4)  # Reshape to [B, C_fourier_spatial, 4, 4]

        # C1: Spatial Conditioning (concat to reshaped Fourier features)
        if self.config_model.stylegan3_g_spatial_cond and spatial_map_g is not None:
            if x.shape[2:] != spatial_map_g.shape[2:]:  # Ensure spatial_map_g is 4x4
                raise ValueError(f"StyleGAN3 G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match initial feature map shape {x.shape} (expect 4x4).")
            x = torch.cat([x, spatial_map_g], dim=1)
            # The first StyleGAN3Layer in self.layers was already initialized with increased in_channels.

        if noise_inputs is None:
            noise_inputs = self.make_noise_sg3(batch_size)

        # Pass through layers
        # StyleGAN3's skip connections and ToRGB logic can be different.
        # Simplified path:
        rgb_out = None  # For progressive ToRGB if used

        for i, layer in enumerate(self.layers):
            noise_i = noise_inputs[i] if i < len(noise_inputs) else None
            x = layer(x, current_w, noise=noise_i)  # Pass the same w to all layers for now

            # Conceptual: if ToRGBs were per layer:
            # current_rgb = self.torgbs[i](x, current_w, skip_rgb=rgb_out)
            # rgb_out = F.interpolate(current_rgb, scale_factor=2) if layer.upsample and i < len(self.layers)-1 else current_rgb
            # For now, only a final ToRGB

        rgb_out = self.final_torgb(x, current_w)  # Pass final features and w to final ToRGB

        return torch.tanh(rgb_out)


class StyleGAN3Discriminator(nn.Module):
    # Placeholder, StyleGAN3 D also has alias-free properties.
    # Can be similar to StyleGAN2 D but with FIR filters and alias-free activations.
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size
        self.channel_multiplier = self.config_model.stylegan3_channel_multiplier
        self.fir_kernel = self.config_model.stylegan3_fir_kernel

        self.channels = {  # Similar to StyleGAN2
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))

        convs = []

        from_rgb_in_channels_sg3 = 3
        if self.config_model.stylegan3_d_spatial_cond:  # C4 for StyleGAN3 D
            from_rgb_in_channels_sg3 += self.config_model.superpixel_spatial_map_channels_d

        # FromRGB layer
        convs.append(EqualizedConv2d(from_rgb_in_channels_sg3, self.channels[self.image_size], 1))
        # convs.append(AliasFreeActivation(negative_slope=0.2, fir_kernel=self.fir_kernel)) # Conceptual

        in_ch = self.channels[self.image_size]
        for i in range(self.log_size, 2, -1):
            out_ch = self.channels[2 ** (i - 1)]
            # Simplified ConvBlock, ideally would use StyleGAN3-specific layers
            # with FIR downsampling and alias-free activations
            convs.append(ConvBlock(in_ch, out_ch, 3, downsample=True,
                                   blur_kernel=self.fir_kernel))  # Using StyleGAN2's ConvBlock
            in_ch = out_ch

        convs.append(ConvBlock(in_ch, self.channels[4], 3))

        final_conv_channels = self.channels[4]
        convs.append(EqualizedConv2d(final_conv_channels, final_conv_channels, 4, padding=0))
        # convs.append(AliasFreeActivation(negative_slope=0.2, fir_kernel=self.fir_kernel))
        convs.append(nn.Flatten())
        convs.append(EqualizedLinear(final_conv_channels, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, image, graph_data=None, spatial_map_d=None):  # New arg
        input_to_d = image
        if self.config_model.stylegan3_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"StyleGAN3 D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)
        return self.convs(input_to_d).squeeze(1)


print("src/models.py created and populated with Generator, Discriminator, and helper modules.")
