import torch
from torch_fidelity.fidelity import calculate_metrics

def calculate_fid(real_images, fake_images, batch_size, cuda=True):
    """
    Calculates the Frechet Inception Distance (FID) between two sets of images.

    Args:
        real_images (torch.Tensor): A tensor of real images.
        fake_images (torch.Tensor): A tensor of fake images.
        batch_size (int): Batch size to use for calculations.
        cuda (bool): Whether to use CUDA.

    Returns:
        float: The calculated FID score.
    """
    # torch-fidelity expects images in range [0, 255] and of type uint8
    real_images = (real_images * 255).byte()
    fake_images = (fake_images * 255).byte()

    metrics_dict = calculate_metrics(
        input1=real_images,
        input2=fake_images,
        cuda=cuda,
        isc=False,
        fid=True,
        kid=False,
        verbose=False,
        batch_size=batch_size,
    )
    return metrics_dict["frechet_inception_distance"]
