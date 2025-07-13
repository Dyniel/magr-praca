import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.blocks import (
    PixelNorm,
    EqualizedLinear,
    ModulatedConv2d,
    NoiseInjection,
    StyleBlock,
    ToRGB,
    EqualizedConv2d,
    ConvBlock,
)

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
        return self.net(z)

class StyleGAN2Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size

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

        self.log_size = int(math.log2(self.image_size))
        self.num_layers_total_for_w = 2 * (self.log_size - 1)

        initial_block_channels = self.channels[4]
        if self.config_model.stylegan2_g_spatial_cond:
            initial_block_channels += self.config_model.superpixel_spatial_map_channels_g

        self.initial_constant = nn.Parameter(torch.randn(1, self.channels[4], 4, 4))
        self.initial_conv = StyleBlock(initial_block_channels, self.channels[4], self.w_dim, num_layers=1)
        self.initial_torgb = ToRGB(self.channels[4], self.w_dim)

        self.blocks = nn.ModuleList()
        self.torgbs = nn.ModuleList()

        in_ch = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_ch = self.channels[2 ** i]
            self.blocks.append(StyleBlock(in_ch, out_ch, self.w_dim, upsample=True))
            self.torgbs.append(ToRGB(out_ch, self.w_dim))
            in_ch = out_ch

        self.blur_kernel_default = [1, 3, 3, 1]
        self.noises_fixed = None

    def w_to_styles(self, w, num_total_layers_for_w):
        if w.ndim == 2:
            return w.unsqueeze(1).repeat(1, num_total_layers_for_w, 1)
        elif w.ndim == 3 and w.shape[1] == num_total_layers_for_w:
            return w
        else:
            raise ValueError(f"w has incompatible shape: {w.shape}")

    def forward(self, z_noise, style_mix_prob=0.9, input_is_w=False,
                truncation_psi=None, truncation_cutoff=None, w_avg=None,
                spatial_map_g=None, z_superpixel_g=None):

        batch_size = z_noise.shape[0] if z_noise is not None else \
            (w_avg.shape[0] if input_is_w and w_avg is not None else
             (spatial_map_g.shape[0] if spatial_map_g is not None else
              (z_superpixel_g.shape[0] if z_superpixel_g is not None else 1)))

        current_z = z_noise
        if self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
            if current_z is None and input_is_w:
                print("Warning (StyleGAN2 G): Latent superpixel conditioning (C2) requested but z_noise is None (input_is_w=True). C2 skipped.")
            elif current_z is not None:
                if current_z.ndim == 4 and current_z.shape[2:] == (1, 1): current_z = current_z.squeeze(-1).squeeze(-1)
                current_z = torch.cat([current_z, z_superpixel_g], dim=1)

        if not input_is_w:
            if current_z is None:
                raise ValueError("StyleGAN2 G: z_noise (current_z) is None and input_is_w is False.")
            w = self.mapping_network(current_z)
        else:
            if current_z is not None and self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
                print("Warning/Info (StyleGAN2 G): input_is_w=True. Assuming z_noise arg is actually W. Latent superpixel conditioning (C2) might not behave as expected if z_noise was not the initial Z.")
                w = current_z
            else:
                w = z_noise

        if truncation_psi is not None and w_avg is not None:
            if w.ndim == 2:
                w_broadcast = w.unsqueeze(1).repeat(1, self.num_layers_total_for_w, 1)
            else:
                w_broadcast = w
            if truncation_cutoff is None:
                w_truncated = w_avg + truncation_psi * (w_broadcast - w_avg)
            else:
                w_前半 = w_avg + truncation_psi * (w_broadcast[:, :truncation_cutoff] - w_avg)
                w_後半 = w_broadcast[:, truncation_cutoff:]
                w_truncated = torch.cat([w_前半, w_後半], dim=1)
            styles = w_truncated
        else:
            if self.training and style_mix_prob > 0 and torch.rand(()).item() < style_mix_prob:
                if input_is_w:
                    print("Warning: StyleGAN2 G received input_is_w=True with style_mix_prob > 0. Style mixing skipped.")
                    styles = self.w_to_styles(w, self.num_layers_total_for_w)
                else:
                    z2 = torch.randn_like(z_noise)
                    if self.config_model.stylegan2_g_latent_cond and z_superpixel_g is not None:
                        z2 = torch.cat([z2, z_superpixel_g], dim=1)
                    w2 = self.mapping_network(z2)
                    mix_cutoff = torch.randint(1, self.num_layers_total_for_w, (1,)).item()
                    w_part1 = w.unsqueeze(1).repeat(1, mix_cutoff, 1)
                    w_part2 = w2.unsqueeze(1).repeat(1, self.num_layers_total_for_w - mix_cutoff, 1)
                    styles = torch.cat([w_part1, w_part2], dim=1)
            else:
                styles = self.w_to_styles(w, self.num_layers_total_for_w)

        x = self.initial_constant.repeat(batch_size, 1, 1, 1)
        if self.config_model.stylegan2_g_spatial_cond and spatial_map_g is not None:
            if x.shape[2:] != spatial_map_g.shape[2:]:
                raise ValueError(f"StyleGAN2 G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match initial constant shape {x.shape} for spatial conditioning.")
            x = torch.cat([x, spatial_map_g], dim=1)

        x = self.initial_conv(x, styles[:, 0])
        rgb = self.initial_torgb(x, styles[:, 1])

        for i, (block, torgb) in enumerate(zip(self.blocks, self.torgbs)):
            style_for_block = styles[:, 2 + 2 * i]
            style_for_torgb = styles[:, 3 + 2 * i]
            x = block(x, style_for_block)
            rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
            rgb = torgb(x, style_for_torgb, skip_rgb=rgb)

        return torch.tanh(rgb)

class StyleGAN2Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size
        self.channel_multiplier = self.config_model.stylegan2_channel_multiplier
        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * self.channel_multiplier,
            128: 128 * self.channel_multiplier, 256: 64 * self.channel_multiplier,
            512: 32 * self.channel_multiplier, 1024: 16 * self.channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))
        self.blur_kernel_default = [1, 3, 3, 1]

        convs = []
        from_rgb_in_channels = 3
        if self.config_model.stylegan2_d_spatial_cond:
            from_rgb_in_channels += self.config_model.superpixel_spatial_map_channels_d

        convs.append(EqualizedConv2d(from_rgb_in_channels, self.channels[self.image_size], 1, activation='lrelu'))

        in_ch = self.channels[self.image_size]
        for i in range(self.log_size, 2, -1):
            out_ch = self.channels[2 ** (i - 1)]
            convs.append(ConvBlock(in_ch, out_ch, 3, downsample=True, blur_kernel=self.blur_kernel_default))
            in_ch = out_ch

        convs.append(ConvBlock(in_ch, self.channels[4], 3))
        final_conv_channels = self.channels[4]
        convs.append(EqualizedConv2d(final_conv_channels, final_conv_channels, 4, padding=0))
        convs.append(nn.Flatten())
        convs.append(EqualizedLinear(final_conv_channels, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, image, graph_data=None, spatial_map_d=None):
        input_to_d = image
        if self.config_model.stylegan2_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"StyleGAN2 D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)
        return self.convs(input_to_d).squeeze(1)
