import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.blocks import WSConv2d

class DCGANGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model

        initial_nz = self.config_model.dcgan_z_dim
        if self.config_model.dcgan_g_latent_cond:
            initial_nz += self.config_model.superpixel_latent_embedding_dim

        ngf = self.config_model.dcgan_g_feat
        nc = 3

        channels_after_first_block = ngf * 8
        if self.config_model.dcgan_g_spatial_cond:
            channels_after_first_block += self.config_model.superpixel_spatial_map_channels_g

        self.main = nn.Sequential(
            nn.ConvTranspose2d(initial_nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels_after_first_block),
            nn.ReLU(True),
            nn.ConvTranspose2d(channels_after_first_block, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        current_size = 32
        current_feat = ngf

        if config.image_size == 64:
            self.main.add_module("conv_transpose_out_64", nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
            self.main.add_module("tanh_out_64", nn.Tanh())
        elif config.image_size == 128:
            self.main.add_module("conv_transpose_64",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))
            self.main.add_module("bn_64", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_64", nn.ReLU(True))
            current_feat //= 2
            self.main.add_module("conv_transpose_out_128",
                                 nn.ConvTranspose2d(current_feat, nc, 4, 2, 1, bias=False))
            self.main.add_module("tanh_out_128", nn.Tanh())
        elif config.image_size == 256:
            self.main.add_module("conv_transpose_64",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))
            self.main.add_module("bn_64", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_64", nn.ReLU(True))
            current_feat //= 2
            current_size = 64
            self.main.add_module("conv_transpose_128",
                                 nn.ConvTranspose2d(current_feat, current_feat // 2, 4, 2, 1, bias=False))
            self.main.add_module("bn_128", nn.BatchNorm2d(current_feat // 2))
            self.main.add_module("relu_128", nn.ReLU(True))
            current_feat //= 2
            current_size = 128
            self.main.add_module("conv_transpose_out_256",
                                 nn.ConvTranspose2d(current_feat, nc, 4, 2, 1, bias=False))
            self.main.add_module("tanh_out_256", nn.Tanh())
        else:
            if config.image_size != 32:
                raise ValueError(
                    f"DCGANGenerator: image_size {config.image_size} not directly supported by this dynamic structure. Base output is 32x32. Add more layers or check config.")
            else:
                self.main.add_module("conv_transpose_out_32",
                                     nn.ConvTranspose2d(ngf, nc, 1, 1, 0, bias=False))
                self.main.add_module("tanh_out_32", nn.Tanh())

    def forward(self, z_noise, graph_batch=None, real_images=None, segments_map=None, adj_matrix=None,
                spatial_map_g=None, z_superpixel_g=None):
        if self.config_model.dcgan_g_latent_cond and z_superpixel_g is not None:
            if z_noise.ndim == 4 and z_noise.shape[2] == 1 and z_noise.shape[3] == 1:
                z_noise = z_noise.squeeze(-1).squeeze(-1)
            elif z_noise.ndim != 2:
                raise ValueError(f"DCGAN G: z_noise has unexpected shape {z_noise.shape} for latent conditioning.")
            if z_superpixel_g.ndim != 2:
                raise ValueError(
                    f"DCGAN G: z_superpixel_g has unexpected shape {z_superpixel_g.shape} for latent conditioning, expected [B, embed_dim].")
            z_noise = torch.cat([z_noise, z_superpixel_g], dim=1)
        if z_noise.ndim == 2:
            z_noise = z_noise.unsqueeze(-1).unsqueeze(-1)

        if self.config_model.dcgan_g_spatial_cond and spatial_map_g is not None:
            x = self.main[0](z_noise)
            if x.shape[2:] != spatial_map_g.shape[2:]:
                raise ValueError(f"DCGAN G: spatial_map_g shape {spatial_map_g.shape} "
                                 f"does not match feature map shape {x.shape} for spatial conditioning at 4x4 stage.")
            x = torch.cat([x, spatial_map_g], dim=1)
            x = self.main[1](x)
            x = self.main[2](x)
            for i in range(3, len(self.main)):
                x = self.main[i](x)
            return x
        else:
            return self.main(z_noise)

class DCGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        ndf = self.config_model.dcgan_d_feat

        initial_input_channels = 3
        if self.config_model.dcgan_d_spatial_cond:
            initial_input_channels += self.config_model.superpixel_spatial_map_channels_d

        self.main = nn.Sequential()
        current_size = config.image_size
        in_c = initial_input_channels

        while current_size > 4:
            out_c = ndf
            if current_size == config.image_size:
                out_c = ndf
            else:
                out_c = in_c * 2 if (in_c * 2) <= (ndf * 8) else (ndf * 8)

            self.main.add_module(f"conv_{current_size}x{current_size}",
                                 nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
            self.main.add_module(f"bn_{current_size}x{current_size}", nn.BatchNorm2d(out_c))
            self.main.add_module(f"lrelu_{current_size}x{current_size}", nn.LeakyReLU(0.2, inplace=True))
            in_c = out_c
            current_size //= 2

        self.main.add_module("conv_final_4x4", nn.Conv2d(in_c, 1, 4, 1, 0, bias=False))

    def forward(self, image, graph_data=None, spatial_map_d=None):
        input_to_d = image
        if self.config_model.dcgan_d_spatial_cond and spatial_map_d is not None:
            if image.shape[2:] != spatial_map_d.shape[2:]:
                raise ValueError(
                    f"DCGAN D: image shape {image.shape} and spatial_map_d shape {spatial_map_d.shape} H,W mismatch.")
            input_to_d = torch.cat([image, spatial_map_d], dim=1)
        return self.main(input_to_d).squeeze()
