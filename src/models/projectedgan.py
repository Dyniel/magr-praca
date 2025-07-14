import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0

from src.models.stylegan2 import StyleGAN2Generator

# A simple feature extractor wrapper
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet50', layers_to_extract=None, pretrained=True, requires_grad=False):
        super().__init__()
        self.model_name = model_name
        if model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
            self.layers_to_extract = layers_to_extract if layers_to_extract is not None else ['layer1', 'layer2', 'layer3']
        elif model_name == 'efficientnet_b0':
            self.model = efficientnet_b0(pretrained=pretrained)
            # Example layers for efficientnet, adjust as needed
            self.layers_to_extract = layers_to_extract if layers_to_extract is not None else ['features.2', 'features.4', 'features.6']
        else:
            raise ValueError(f"Unsupported feature extractor model: {model_name}")

        self.hooks = {}
        self.features = {}

        for name, module in self.model.named_modules():
            if name in self.layers_to_extract:
                self.hooks[name] = module.register_forward_hook(self._get_hook(name))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _get_hook(self, name):
        def hook(model, input, output):
            self.features[name] = output
        return hook

    def forward(self, x):
        self.features.clear()
        _ = self.model(x)
        return self.features

    def __del__(self):
        for hook in self.hooks.values():
            hook.remove()

class ProjectedGANGenerator(StyleGAN2Generator):
    # Inherits from StyleGAN2Generator, no changes needed for basic Projected GAN
    pass

class ProjectedGANDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Using a simple discriminator for now, this can be replaced with a more complex one
        self.main = nn.Sequential(
            nn.Conv2d(config.model.superpixel_spatial_map_channels_d + 3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img, **kwargs):
        return self.main(img)
