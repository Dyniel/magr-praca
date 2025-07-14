import torch
import torch.nn.functional as F

def generator_loss_nonsaturating(d_fake_logits):
    return F.softplus(-d_fake_logits).mean()

def discriminator_loss_r1(d_real_logits, d_fake_logits):
    return F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()

def generator_loss_bce(d_fake_logits):
    return F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))

def discriminator_loss_bce(d_real_logits, d_fake_logits):
    loss_real = F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits))
    loss_fake = F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
    return loss_real + loss_fake

def r1_penalty(d_real_logits, real_images, r1_gamma):
    if r1_gamma == 0:
        return torch.tensor(0.0, device=real_images.device)

    grad_real = torch.autograd.grad(
        outputs=d_real_logits.sum(), inputs=real_images, create_graph=True
    )[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    if torch.isnan(grad_penalty):
        # This can happen, for example, if the gradients themselves are NaN.
        # Returning a zero tensor with requires_grad=False to avoid disrupting the graph.
        print("Warning: grad_penalty resulted in NaN. Returning 0.")
        return torch.tensor(0.0, device=real_images.device, requires_grad=False)

    return grad_penalty * (r1_gamma / 2)

# WGAN-GP losses
def generator_loss_wgan(d_fake_logits):
    """
    Generator loss for WGAN-GP.
    It tries to maximize the score of fake images.
    """
    return -d_fake_logits.mean()

def discriminator_loss_wgan(d_real_logits, d_fake_logits):
    """
    Discriminator loss for WGAN-GP.
    It tries to maximize the difference between the scores of real and fake images.
    """
    return d_fake_logits.mean() - d_real_logits.mean()

def gradient_penalty(discriminator, real, fake, device):
    α = torch.rand(real.size(0), 1, 1, 1, device=device)
    x_hat = (α * real + (1-α) * fake).requires_grad_(True)
    d_hat = discriminator(x_hat)

    grads = torch.autograd.grad(
        outputs=d_hat.sum(), inputs=x_hat, create_graph=True
    )[0].view(real.size(0), -1)
    grad_norm = grads.norm(2, dim=1)

    # karz tylko, gdy ||∇|| > 1
    penalty = ((grad_norm - 1).clamp(min=0) ** 2).mean()
    return penalty

def generator_loss_hinge(d_fake_logits):
    return -d_fake_logits.mean()

def discriminator_loss_hinge(d_real_logits, d_fake_logits):
    real_loss = F.relu(1.0 - d_real_logits).mean()
    fake_loss = F.relu(1.0 + d_fake_logits).mean()
    return real_loss + fake_loss
