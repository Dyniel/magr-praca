import torch
import torch.nn as nn

class HistogramLoss(nn.Module):
    def __init__(self, bins=64, loss_type='l1', value_range=(0, 1)):
        super().__init__()
        self.bins = bins
        self.loss_type = loss_type
        self.value_range = value_range

    def _calculate_histogram(self, x):
        # Normalize to [0, 1] for histogram calculation
        x_norm = (x - self.value_range[0]) / (self.value_range[1] - self.value_range[0])
        x_norm = x_norm.clamp(0, 1)

        # Reshape to (B*C, H*W) to calculate histogram per channel
        B, C, H, W = x_norm.shape
        x_flat = x_norm.view(B * C, -1)

        hists = []
        for i in range(x_flat.shape[0]):
            hist = torch.histc(x_flat[i], bins=self.bins, min=0, max=1)
            hists.append(hist)

        # Stack and normalize histograms
        batch_hist = torch.stack(hists, dim=0)
        batch_hist = batch_hist / batch_hist.sum(dim=1, keepdim=True).clamp(min=1e-8) # Normalize each histogram
        return batch_hist.view(B, C, self.bins)

    def forward(self, fake_images, real_images):
        hist_fake = self._calculate_histogram(fake_images)
        hist_real = self._calculate_histogram(real_images)

        if self.loss_type == 'l1':
            return torch.abs(hist_fake - hist_real).mean()
        elif self.loss_type == 'l2':
            return torch.pow(hist_fake - hist_real, 2).mean()
        elif self.loss_type == 'cosine':
            return 1 - F.cosine_similarity(hist_fake, hist_real, dim=-1).mean()
        else:
            raise ValueError(f"Unsupported histogram loss type: {self.loss_type}")
