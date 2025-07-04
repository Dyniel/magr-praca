import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils # For making image grids

# Local imports
from src.models import Generator, Discriminator
from src.data_loader import get_dataloader
from src.utils import (
    save_checkpoint, load_checkpoint, setup_wandb, log_to_wandb,
    denormalize_image # To convert [-1,1] to [0,1] for logging images
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create output directories
        os.makedirs(self.config.output_dir_run, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.config.output_dir_run, "checkpoints")
        self.samples_dir = os.path.join(self.config.output_dir_run, "samples")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        # Reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True # Can slow down, but good for reproducibility
            torch.backends.cudnn.benchmark = False   # Disable benchmark mode for determinism

        # Initialize Models
        self.G = Generator(config).to(self.device)
        self.D = Discriminator(config).to(self.device)

        # Initialize Optimizers
        self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
        self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))

        # Initialize DataLoader
        self.dataloader = get_dataloader(config, shuffle=True)

        # A fixed batch for generating samples during training
        self.fixed_sample_batch = self._prepare_fixed_sample_batch()

        # Initialize Weights & Biases (if enabled)
        self.wandb_run = setup_wandb(config, self.G, project_name=config.wandb_project_name, watch_model=True)

        self.current_epoch = 0
        self.current_step = 0
        # TODO: Implement checkpoint loading if resume is enabled

    def _prepare_fixed_sample_batch(self):
        """Prepares a fixed batch of data for consistent sample generation."""
        try:
            fixed_batch = next(iter(self.dataloader))
            # Keep only the number of samples needed, move to device
            num_samples = self.config.num_samples_to_log
            return {
                "image": fixed_batch["image"][:num_samples].to(self.device),
                "segments": fixed_batch["segments"][:num_samples].to(self.device),
                "adj": fixed_batch["adj"][:num_samples].to(self.device)
            }
        except Exception as e:
            print(f"Could not prepare fixed sample batch: {e}. Sample generation might be affected.")
            return None

    def _relativistic_loss_d(self, d_real, d_fake):
        """Relativistic average Standard GAN loss for Discriminator."""
        # Real images should be more 'real' than fake images
        errD_real = F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.ones_like(d_real))
        # Fake images should be less 'real' than real images
        errD_fake = F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.zeros_like(d_fake))
        return (errD_real + errD_fake) / 2.0
        # Alternative from gan5 (based on softplus, similar to RaLSGAN):
        # return (F.softplus(-d_real + d_fake.mean()) + F.softplus(d_fake - d_real.mean())).mean()


    def _relativistic_loss_g(self, d_real, d_fake):
        """Relativistic average Standard GAN loss for Generator."""
        # Generator wants fake images to be more 'real' than real images
        errG_real = F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), torch.zeros_like(d_real))
        # Generator wants fake images to BE real
        errG_fake = F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), torch.ones_like(d_fake))
        return (errG_real + errG_fake) / 2.0
        # Alternative from gan5 (based on simple -D(fake).mean() if not relativistic):
        # return -d_fake.mean() # This is for standard GAN, not RaLSGAN.
        # For RaLSGAN (from original gan5 which used rel_loss for D, but -D(fake).mean() for G):
        # The G loss in gan5 was simply -self.D(fake).mean()
        # If using relativistic D, G should also be relativistic for consistency.
        # The provided D loss in gan5: (F.softplus(-r + f) + F.softplus(f - r)).mean()
        # The G loss was -f_scr.mean().
        # Let's use the gan5 D loss for now, and the G loss from gan5.

    def _r1_gradient_penalty(self, real_images, d_real_logits):
        """Calculates R1 gradient penalty."""
        grad_real = torch.autograd.grad(
            outputs=d_real_logits.sum(), inputs=real_images, create_graph=True
        )[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        return grad_penalty

    def train_epoch(self):
        self.G.train()
        self.D.train()

        loop = tqdm(self.dataloader, desc=f"Epoch [{self.current_epoch}/{self.config.num_epochs}]")
        total_g_loss = 0.0
        total_d_loss = 0.0

        for batch_idx, batch_data in enumerate(loop):
            real_images = batch_data["image"].to(self.device)
            segments_map = batch_data["segments"].to(self.device)
            adj_matrix = batch_data["adj"].to(self.device)

            current_batch_size = real_images.size(0)

            # --- Train Discriminator ---
            # According to gan5: d_updates_per_g_update (e.g., 2 D steps per G step)
            # For simplicity in this refactor, let's do 1 D step then 1 G step.
            # If d_updates_per_g_update > 1 is desired, this loop structure needs adjustment.
            # For now, assuming d_updates_per_g_update = 1 for trainer structure.
            # The config has d_updates_per_g_update, so let's implement it.

            for _ in range(self.config.d_updates_per_g_update):
                self.optD.zero_grad()

                # Latent vectors for G
                z = torch.randn(current_batch_size, self.config.z_dim, device=self.device)

                with torch.no_grad(): # Detach G's output when training D
                    fake_images = self.G(z, real_images, segments_map, adj_matrix)

                d_real_logits = self.D(real_images)
                d_fake_logits = self.D(fake_images) # Use detached fakes

                # Relativistic loss from gan5 for D
                # lossD_relativistic = (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean() # This is RaHingeGAN
                # Original gan5 D loss: self.rel_loss(r_scr, f_scr) -> (F.softplus(-r + f) + F.softplus(f - r)).mean()
                # This is (F.softplus(-d_real_logits + d_fake_logits.mean()) + F.softplus(d_fake_logits - d_real_logits.mean())).mean() if using .mean() for average scores
                # Or per-sample comparison: (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()
                # Let's use the per-sample comparison version from gan5's rel_loss
                lossD_adv = (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()


                # R1 Gradient Penalty
                real_images.requires_grad_(True) # Enable gradient for penalty calculation
                d_real_logits_for_gp = self.D(real_images) # Re-compute D output for penalty
                r1_penalty = self._r1_gradient_penalty(real_images, d_real_logits_for_gp)
                real_images.requires_grad_(False) # Disable again

                lossD = lossD_adv + self.config.r1_gamma * 0.5 * r1_penalty # 0.5 factor for R1 is common

                lossD.backward()
                self.optD.step()

            # --- Train Generator ---
            self.optG.zero_grad()

            z = torch.randn(current_batch_size, self.config.z_dim, device=self.device)
            # We need fresh fake images with gradients flowing back to G
            fake_images_for_g = self.G(z, real_images, segments_map, adj_matrix)
            d_fake_for_g_logits = self.D(fake_images_for_g)

            # Generator loss from gan5: -d_fake_for_g_logits.mean()
            lossG = -d_fake_for_g_logits.mean()

            lossG.backward()
            self.optG.step()

            total_d_loss += lossD.item()
            total_g_loss += lossG.item()

            if self.current_step % self.config.log_freq_step == 0:
                log_data = {
                    "Epoch": self.current_epoch,
                    "Step": self.current_step,
                    "Loss_D": lossD.item(),
                    "Loss_D_Adv": lossD_adv.item(),
                    "R1_Penalty": r1_penalty.item(),
                    "Loss_G": lossG.item(),
                    "D_Real_Logits_Mean": d_real_logits.mean().item(),
                    "D_Fake_Logits_Mean": d_fake_logits.mean().item(),
                }
                log_to_wandb(self.wandb_run, log_data, step=self.current_step)
                loop.set_postfix(log_data)

            self.current_step +=1

        avg_d_loss = total_d_loss / len(self.dataloader)
        avg_g_loss = total_g_loss / len(self.dataloader)
        print(f"Epoch {self.current_epoch} finished. Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
        log_to_wandb(self.wandb_run, {"Epoch_Avg_D_Loss": avg_d_loss, "Epoch_Avg_G_Loss": avg_g_loss}, step=self.current_step)


    def generate_samples(self, epoch, step):
        if not self.fixed_sample_batch:
            print("Fixed sample batch not available. Skipping sample generation.")
            return

        self.G.eval()
        with torch.no_grad():
            z_sample = torch.randn(self.fixed_sample_batch["image"].size(0), self.config.z_dim, device=self.device)

            generated_samples = self.G(
                z_sample,
                self.fixed_sample_batch["image"],
                self.fixed_sample_batch["segments"],
                self.fixed_sample_batch["adj"]
            )

            # Denormalize from [-1,1] to [0,1] for saving/logging
            generated_samples = denormalize_image(generated_samples)
            real_samples_for_log = denormalize_image(self.fixed_sample_batch["image"])

            # Create grid
            img_grid_fake = vutils.make_grid(generated_samples, normalize=False, nrow=int(np.sqrt(generated_samples.size(0))))
            img_grid_real = vutils.make_grid(real_samples_for_log, normalize=False, nrow=int(np.sqrt(real_samples_for_log.size(0))))

            # Save to file
            vutils.save_image(img_grid_fake, os.path.join(self.samples_dir, f"fake_samples_epoch_{epoch:04d}_step_{step}.png"))
            if epoch == 0 and step <= self.config.log_freq_step *2 : # Save real samples once at the beginning
                 vutils.save_image(img_grid_real, os.path.join(self.samples_dir, f"real_samples_epoch_{epoch:04d}.png"))


            # Log to WandB
            if self.wandb_run:
                log_images_wandb = {"Generated_Samples_Epoch_{}".format(epoch): self.wandb_run.Image(img_grid_fake)}
                if epoch == 0 and step <= self.config.log_freq_step *2:
                     log_images_wandb["Real_Samples"] = self.wandb_run.Image(img_grid_real)
                log_to_wandb(self.wandb_run, log_images_wandb, step=step)
        self.G.train()


    def train(self):
        print("Starting training...")
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()

            if epoch % self.config.sample_freq_epoch == 0:
                self.generate_samples(epoch, self.current_step)

            if epoch % self.config.checkpoint_freq_epoch == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch:04d}.pth.tar")
                save_checkpoint({
                    'epoch': epoch,
                    'step': self.current_step,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                    'optG_state_dict': self.optG.state_dict(),
                    'optD_state_dict': self.optD.state_dict(),
                    'config': self.config # Save config with checkpoint
                }, is_best=False, filename=checkpoint_path) # 'is_best' logic can be added if there's a validation metric
                print(f"Checkpoint saved to {checkpoint_path}")

        print("Training finished.")
        if self.wandb_run:
            self.wandb_run.finish()

print("src/trainer.py created with Trainer class.")
