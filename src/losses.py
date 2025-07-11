import torch
import torch.nn.functional as F

def compute_histogram(images_batch, bins=256, min_val=0.0, max_val=1.0, normalize=True):
    """
    Computes channel-wise histograms for a batch of images.
    Assumes images are in [B, C, H, W] format and values are in [min_val, max_val].
    This is a simplified version using torch.histogram, which might have limitations
    for direct backpropagation if bin edges are not constant or if the operation
    itself isn't perfectly differentiable for all downstream uses.
    A "soft histogram" would be more robust for training if issues arise.

    Args:
        images_batch (Tensor): Batch of images, shape [B, C, H, W]. Expected to be in [0,1] range if default min/max.
        bins (int): Number of bins for the histogram.
        min_val (float): Minimum value for histogram range.
        max_val (float): Maximum value for histogram range.
        normalize (bool): If True, normalize histograms to sum to 1 (probability distribution).

    Returns:
        Tensor: Batch of histograms, shape [B, C, bins].
    """
    batch_size, num_channels, _, _ = images_batch.shape
    histograms = []

    for i in range(batch_size):
        img_histograms = []
        for c in range(num_channels):
            channel_data = images_batch[i, c, :, :].flatten()
            # torch.histogram requires 1D input
            hist = torch.histc(channel_data, bins=bins, min=min_val, max=max_val)
            if normalize:
                hist = hist / torch.sum(hist).clamp(min=1e-12) # Avoid division by zero
            img_histograms.append(hist)
        histograms.append(torch.stack(img_histograms)) # Shape [C, bins]

    batch_histograms = torch.stack(histograms) # Shape [B, C, bins]
    return batch_histograms

class HistogramLoss(torch.nn.Module):
    def __init__(self, bins=256, loss_type='l1', value_range=(0.0, 1.0)):
        """
        Loss based on the difference between color histograms of real and fake images.
        Args:
            bins (int): Number of bins for histogram calculation.
            loss_type (str): Type of loss to use ('l1', 'l2', 'cosine').
            value_range (tuple): Min and max values of input images (e.g., (0.0, 1.0) or (-1.0, 1.0)).
                                 Images will be normalized to [0,1] for histogram computation if not already.
        """
        super().__init__()
        self.bins = bins
        self.loss_type = loss_type
        self.value_range_min = value_range[0]
        self.value_range_max = value_range[1]

        if loss_type == 'l1':
            self.distance_metric = F.l1_loss
        elif loss_type == 'l2':
            self.distance_metric = F.mse_loss
        elif loss_type == 'cosine':
            self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x.view(x.shape[0],-1), y.view(y.shape[0],-1)).mean()
            # Cosine similarity needs flattening of C,bins dimensions or careful application.
            # For simplicity, let's apply it channel-wise and average, or flatten.
            # The lambda above flattens [B,C,bins] to [B, C*bins] for cosine sim.
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'l1', 'l2', or 'cosine'.")

    def _normalize_images_for_hist(self, images):
        # Shift and scale images to [0, 1] range for histogram computation
        # if they are not already in that range (e.g. if they are in [-1, 1])
        if self.value_range_min == -1.0 and self.value_range_max == 1.0: # Common GAN output range
            return (images + 1.0) / 2.0
        elif self.value_range_min == 0.0 and self.value_range_max == 1.0:
            return images
        else:
            # General case, though less common for image tensors directly
            # This might need adjustment if the input range isn't one of the two above.
            # Forcing clamp just in case, though ideally input should match value_range.
            images_clamped = torch.clamp(images, self.value_range_min, self.value_range_max)
            return (images_clamped - self.value_range_min) / (self.value_range_max - self.value_range_min)


    def forward(self, fake_images, real_images):
        """
        Args:
            fake_images (Tensor): Batch of generated images, shape [B, C, H, W].
            real_images (Tensor): Batch of real images, shape [B, C, H, W].
        """
        # Normalize images to [0,1] for histogram computation
        fake_images_norm = self._normalize_images_for_hist(fake_images)
        real_images_norm = self._normalize_images_for_hist(real_images)

        hist_fake = compute_histogram(fake_images_norm, bins=self.bins, min_val=0.0, max_val=1.0, normalize=True)
        hist_real = compute_histogram(real_images_norm, bins=self.bins, min_val=0.0, max_val=1.0, normalize=True)

        # Average histograms over the batch dimension
        # This is one way; another is to compute loss per sample and then average.
        # HistoGAN paper might specify averaging histograms first.
        avg_hist_fake = torch.mean(hist_fake, dim=0) # Shape [C, bins]
        avg_hist_real = torch.mean(hist_real, dim=0) # Shape [C, bins]

        if self.loss_type == 'cosine':
            # For cosine, distance_metric expects [B, Features] or [Features]
            # Here avg_hist_fake/real are [C, bins]. We can treat each channel separately
            # or flatten them. Let's flatten C*bins.
            loss = self.distance_metric(avg_hist_fake.view(1, -1), avg_hist_real.view(1,-1))
        else: # l1, l2
            loss = self.distance_metric(avg_hist_fake, avg_hist_real)

        return loss

print("src/losses.py created with compute_histogram and HistogramLoss.")
