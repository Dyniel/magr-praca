import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils import spectral_norm

class WSConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, use_spectral_norm=True):
        super().__init__()
        conv_layer = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        if use_spectral_norm:
            self.conv = spectral_norm(conv_layer)
        else:
            self.conv = conv_layer

        negative_slope = 0.2
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        fan_in = in_ch * kernel_size * kernel_size
        std = gain / math.sqrt(fan_in)
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)

        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)

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

        self.modulation = EqualizedLinear(style_dim, in_channels, bias=True)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        if down:
            factor = 2
            p = (factor - 1) + (kernel_size - 1)
            pad_val = p // 2
            self.blur = Blur(blur_kernel, pad=(pad_val, pad_val), downsample_factor=factor)

        self.padding = kernel_size // 2

    def forward(self, x, style):
        batch_size, in_channels, height, width = x.shape
        style = self.modulation(style).view(batch_size, 1, in_channels, 1, 1)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch_size * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )

        if self.down:
            x = self.blur(x)
            x = x.view(1, batch_size * in_channels, x.shape[-2], x.shape[-1])
            out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
            out = out.view(batch_size, self.out_channels, out.shape[-2], out.shape[-1])
        else:
            x = x.reshape(1, batch_size * in_channels, height, width)
            out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
            out = out.view(batch_size, self.out_channels, out.shape[-2], out.shape[-1])

        out = out + self.bias
        return out

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)
        return x + self.weight * noise

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, downsample_factor=1):
        super().__init__()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        if kernel.ndim == 1:
            kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel[None, None, :, :])
        self.pad = pad
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor

    def forward(self, x):
        x = F.pad(x, (self.pad[0], self.pad[1], self.pad[0], self.pad[1]), mode='reflect')
        num_channels = x.size(1)
        kernel_expanded = self.kernel.expand(num_channels, -1, -1, -1)
        if self.upsample_factor > 1:
            x = F.conv_transpose2d(x, kernel_expanded, stride=self.upsample_factor, padding=0, groups=num_channels)
        elif self.downsample_factor > 1:
            x = F.conv2d(x, kernel_expanded, stride=self.downsample_factor, padding=0, groups=num_channels)
        else:
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
                    up=False,
                    blur_kernel=blur_kernel
                )
            )
            self.noises.append(NoiseInjection(out_channel))
            self.activations.append(nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x, style):
        out = x
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        for i in range(self.num_layers):
            out = self.convs[i](out, style)
            out = self.noises[i](out, noise=None)
            out = self.activations[i](out)
        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, out_channels=3, kernel_size=1):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channels, kernel_size, style_dim, demodulate=False)

    def forward(self, x, style, skip_rgb=None):
        rgb = self.conv(x, style)
        if skip_rgb is not None:
            rgb = rgb + skip_rgb
        return rgb

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

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1],
                 activation='lrelu'):
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channel, in_channel, kernel_size, padding=kernel_size // 2,
                                     activation=activation)
        if downsample:
            self.blur = Blur(blur_kernel, pad=((len(blur_kernel) - 1) // 2, (len(blur_kernel) - 1) // 2),
                             downsample_factor=2)
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
            x = self.blur(x)
        x = self.conv2(x)
        return x
