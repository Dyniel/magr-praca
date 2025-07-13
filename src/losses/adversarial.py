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

    print(f"Grad Real: min={grad_real.min().item():.4f}, max={grad_real.max().item():.4f}, mean={grad_real.mean().item():.4f}")

    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    if torch.isnan(grad_penalty):
        print("Grad penalty is nan. Returning 0.")
        return torch.tensor(0.0, device=real_images.device)

    return grad_penalty * (r1_gamma / 2)
