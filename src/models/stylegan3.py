import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.blocks import (
    EqualizedLinear,
    NoiseInjection,
    Blur,
    ToRGB,
    EqualizedConv2d,
    ConvBlock,
)
from src.models.stylegan2 import MappingNetwork


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0, learnable=False):
        super().__init__()
        self.linear = EqualizedLinear(in_features, out_features, lr_mul=1.0)
        self.out_features = out_features

    def forward(self, w):
        fourier_basis = self.linear(w)
        return fourier_basis


class AliasFreeActivation(nn.Module):
    def __init__(self, negative_slope=0.2, upsample_factor=2, downsample_factor=1, fir_kernel=None):
        super().__init__()
        self.negative_slope = negative_slope
        self.activation = nn.LeakyReLU(negative_slope, inplace=True)
        self.use_fir = fir_kernel is not None
        if self.use_fir:
            pad_size = (len(fir_kernel) - 1) // 2
            self.fir_filter_up = Blur(fir_kernel, pad=(pad_size, pad_size),
                                      upsample_factor=upsample_factor) if upsample_factor > 1 else None
            self.fir_filter_down = Blur(fir_kernel, pad=(pad_size, pad_size),
                                        downsample_factor=downsample_factor) if downsample_factor > 1 else None

    def forward(self, x):
        x = self.activation(x)
        return x


class StyleGAN3Layer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=3, upsample=False, fir_kernel=None):
        super().__init__()
        self.upsample = upsample
        self.style_affine = EqualizedLinear(w_dim, in_channels, bias=True)
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size,
                                    padding=kernel_size // 2)
        self.noise_injection = NoiseInjection(out_channels)
        self.activation = AliasFreeActivation(negative_slope=0.2, upsample_factor=(2 if upsample else 1),
                                              fir_kernel=fir_kernel)
        if upsample:
            self.upsampler = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.upsampler = None

    def forward(self, x, w_style, noise=None):
        style = self.style_affine(w_style).unsqueeze(2).unsqueeze(3)
        if self.upsampler:
            x = self.upsampler(x)
        x = self.conv(x * (style + 1))
        x = self.noise_injection(x, noise=None)
        x = self.activation(x)
        return x


class StyleGAN3Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size

        actual_z_dim_for_mapping = self.config_model.stylegan3_z_dim
        if self.config_model.stylegan3_g_latent_cond:
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
        self.num_styles_needed = 2 * (self.log_size - 1)

        self.fourier_features_dim = self.channels[4] * 4 * 4
        self.fourier_input = FourierFeatures(self.w_dim, self.fourier_features_dim)

        self.initial_reshape_channels = self.channels[4]
        first_layer_in_channels = self.initial_reshape_channels
        if self.config_model.stylegan3_g_spatial_cond:
            first_layer_in_channels += self.config_model.superpixel_spatial_map_channels_g

        self.layers = nn.ModuleList()
        self.torgbs = nn.ModuleList()

        in_ch = self.initial_reshape_channels
        current_res = 4

        self.layers.append(
            StyleGAN3Layer(first_layer_in_channels, self.channels[current_res], self.w_dim, fir_kernel=self.fir_kernel))
        in_ch = self.channels[current_res]

        for i in range(3, self.log_size + 1):
            target_res = 2 ** i
            out_ch = self.channels[target_res]
            self.layers.append(StyleGAN3Layer(in_ch, out_ch, self.w_dim, upsample=True, fir_kernel=self.fir_kernel))
            in_ch = out_ch

        self.final_torgb = ToRGB(in_ch, self.w_dim)

    def make_noise_sg3(self, batch_size):
        noises = []
        current_res = 4
        noises.append(
            torch.randn(batch_size, 1, current_res, current_res, device=self.mapping_network.net[1].weight.device))
        for i in range(3, self.log_size + 1):
            current_res *= 2
            noises.append(
                torch.randn(batch_size, 1, current_res, current_res, device=self.mapping_network.net[1].weight.device))
        return noises

    def forward(self, z_noise, noise_inputs=None, input_is_w=False, truncation_psi=None, w_avg=None,
                spatial_map_g=None, z_superpixel_g=None):

        if z_noise is not None:
            batch_size = z_noise.shape[0]
        elif input_is_w and w_avg is not None:
            batch_size = w_avg.shape[0]
        elif input_is_w and z_noise is None:
            if isinstance(spatial_map_g, torch.Tensor):
                batch_size = spatial_map_g.shape[0]
            elif isinstance(z_superpixel_g, torch.Tensor):
                batch_size = z_superpixel_g.shape[0]
            else:
                batch_size = 1
        elif isinstance(spatial_map_g, torch.Tensor):
            batch_size = spatial_map_g.shape[0]
        elif isinstance(z_superpixel_g, torch.Tensor):
            batch_size = z_superpixel_g.shape[0]
        else:
            batch_size = 1

        current_z = z_noise
        if self.config_model.stylegan3_g_latent_cond and z_superpixel_g is not None:
            if current_z is None and input_is_w:
                print("Warning (StyleGAN3 G): Latent superpixel conditioning (C2) requested but z_noise is None (input_is_w=True). C2 skipped.")
            elif current_z is not None:
                if current_z.ndim == 4 and current_z.shape[2:] == (1, 1): current_z = current_z.squeeze(-1).squeeze(-1)
                current_z = torch.cat([current_z, z_superpixel_g], dim=1)

        if not input_is_w:
            if current_z is None:
                raise ValueError("StyleGAN3 G: z_noise (current_z) is None and input_is_w is False.")
            w = self.mapping_network(current_z)
        else:
            if current_z is not None and self.config_model.stylegan3_g_latent_cond and z_superpixel_g is not None:
                print("Warning/Info (StyleGAN3 G): input_is_w=True. Assuming z_noise arg is W. C2 might not behave as expected.")
                w = current_z
            else:
                w = z_noise

        current_w = w
        if w.ndim == 3 and w.shape[1] == 1:
            current_w = w.squeeze(1)
        elif w.ndim == 3 and w.shape[1] > 1:
            print("Warning: StyleGAN3 G received multiple w vectors per sample, using w[:,0] for all layers.")
            current_w = w[:, 0, :]

        if truncation_psi is not None and w_avg is not None:
            current_w = w_avg + truncation_psi * (current_w - w_avg)

        x = self.fourier_input(current_w)
        x = x.view(batch_size, self.initial_reshape_channels, 4, 4)

        if self.config_model.stylegan3_g_spatial_cond and spatial_map_g is not None:
            if x.shape[2:] != spatial_map_g.shape[2:]:
                raise ValueError(f"StyleGAN3 G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match initial feature map shape {x.shape} (expect 4x4).")
            x = torch.cat([x, spatial_map_g], dim=1)

        if noise_inputs is None:
            noise_inputs = self.make_noise_sg3(batch_size)

        rgb_out = None

        for i, layer in enumerate(self.layers):
            noise_i = noise_inputs[i] if i < len(noise_inputs) else None
            x = layer(x, current_w, noise=noise_i)

        rgb_out = self.final_torgb(x, current_w)

        return torch.tanh(rgb_out)


class StyleGAN3Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size
        self.channel_multiplier = self.config_model.stylegan3_channel_multiplier
        self.fir_kernel = self.config_model.stylegan3_fir_kernel

        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))

        convs = []

        from_rgb_in_channels_sg3 = 3
        if self.config_model.stylegan3_d_spatial_cond:
            from_rgb_in_channels_sg3 += self.config_model.superpixel_spatial_map_channels_d

        convs.append(EqualizedConv2d(from_rgb_in_channels_sg3, self.channels[self.image_size], 1))

        in_ch = self.channels[self.image_size]
        for i in range(self.log_size, 2, -1):
            out_ch = self.channels[2 ** (i - 1)]
            convs.append(ConvBlock(in_ch, out_ch, 3, downsample=True,
                                   blur_kernel=self.fir_kernel))
            in_ch = out_ch

        convs.append(ConvBlock(in_ch, self.channels[4], 3))
        in_ch = self.channels[4]

        convs.append(nn.Flatten())
        linear_in_features = in_ch * 4 * 4
        convs.append(EqualizedLinear(linear_in_features, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, image, graph_data=None, spatial_map_d=None):
        input_to_d = image
        if self.config_model.stylegan3_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"StyleGAN3 D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)
        return self.convs(input_to_d).squeeze()
