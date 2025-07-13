import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models.feature_extraction import create_feature_extractor

from src.models.stylegan2 import StyleGAN2Generator
from src.models.blocks import EqualizedConv2d, ConvBlock, EqualizedLinear


class FeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet50", layers_to_extract=None, pretrained=True, requires_grad=False):
        super().__init__()
        if model_name == "resnet50":
            model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            if layers_to_extract is None:
                layers_to_extract = {
                    'relu': 'stem_relu',
                    'layer1': 'layer1_out',
                    'layer2': 'layer2_out',
                    'layer3': 'layer3_out',
                    'layer4': 'layer4_out',
                }
        elif model_name == "efficientnet_b0":
            model = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            if layers_to_extract is None:
                layers_to_extract = {
                    'features.0': 'eff_feat0',
                    'features.2': 'eff_feat1',
                    'features.3': 'eff_feat2',
                    'features.5': 'eff_feat3',
                    'features.8': 'eff_feat_final_conv',
                }
        else:
            raise ValueError(f"Unsupported feature_extractor model_name: {model_name}")

        self.feature_extractor = create_feature_extractor(model, return_nodes=layers_to_extract)

        if not requires_grad:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor.eval()

    def forward(self, x):
        return self.feature_extractor(x)


class ProjectedGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config_model = config.model
        self.image_size = config.image_size

        self.d_channel_multiplier = self.config_model.projectedgan_d_channel_multiplier
        self.d_channels = {
            4: 512, 8: 512, 16: 256 * self.d_channel_multiplier, 32: 128 * self.d_channel_multiplier,
            64: 64 * self.d_channel_multiplier, 128: 32 * self.d_channel_multiplier,
            256: 16 * self.d_channel_multiplier,
        }
        self.log_size = int(math.log2(self.image_size))

        d_convs = []
        pgd_from_rgb_in_channels = 3
        if self.config_model.projectedgan_d_spatial_cond:
            pgd_from_rgb_in_channels += self.config_model.superpixel_spatial_map_channels_d

        d_convs.append(
            EqualizedConv2d(pgd_from_rgb_in_channels, self.d_channels[self.image_size], 1, activation='lrelu'))

        in_ch = self.d_channels[self.image_size]
        for i in range(self.log_size, 2, -1):
            out_ch = self.d_channels[2 ** (i - 1)]
            d_convs.append(
                ConvBlock(in_ch, out_ch, 3, downsample=True, blur_kernel=self.config_model.projectedgan_blur_kernel))
            in_ch = out_ch

        d_convs.append(ConvBlock(in_ch, self.d_channels[4], 3))
        self.d_cnn_path = nn.Sequential(*d_convs)

        self.final_d_conv = EqualizedConv2d(self.d_channels[4], self.d_channels[4], 4, padding=0,
                                            activation='lrelu')
        self.final_d_flatten = nn.Flatten()
        self.final_d_linear = EqualizedLinear(self.d_channels[4], 1)

    def forward(self, image_for_d_path, spatial_map_d=None):
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
        super().__init__(config)
        print("Initialized ProjectedGANGenerator (based on StyleGAN2Generator).")
