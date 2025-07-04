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
import shutil # For FID image directory handling
from PIL import Image # For saving images for FID

# Attempt to import pytorch_fid, if not available, FID calculation will be disabled.
try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    calculate_fid_given_paths = None
    print("Warning: pytorch-fid not found. FID calculation will be disabled. "
          "Install with: pip install pytorch-fid")


class Trainer:
    def __init__(self, config):
        self.config = config
        if calculate_fid_given_paths is None:
            self.config.enable_fid_calculation = False # Disable if library not found
            print("FID calculation has been disabled because pytorch-fid is not installed.")
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

        self.fid_temp_real_path = os.path.join(self.config.output_dir_run, "fid_real_images_temp")
        self.fid_temp_fake_path = os.path.join(self.config.output_dir_run, "fid_fake_images_temp")

        if self.config.resume_checkpoint_path and os.path.exists(self.config.resume_checkpoint_path):
            self.load_training_checkpoint(self.config.resume_checkpoint_path)
        else:
            if self.config.resume_checkpoint_path: # Path given but not found
                 print(f"Warning: resume_checkpoint_path '{self.config.resume_checkpoint_path}' not found. Starting from scratch.")
            self.current_epoch = 0
            self.current_step = 0


    def load_training_checkpoint(self, checkpoint_path):
        """Loads G, D, optimizers, epoch, and step from a checkpoint."""
        start_epoch, current_step = load_checkpoint(
            checkpoint_path,
            self.G, self.D,
            self.optG, self.optD,
            self.device
        )
        # The trainer loop starts from self.current_epoch, so if a checkpoint was saved
        # at the end of epoch N (meaning N epochs completed), we want to start the next epoch, N+1.
        # load_checkpoint returns the epoch number that was *completed*.
        self.current_epoch = start_epoch +1
        self.current_step = current_step
        print(f"Resumed from checkpoint. Starting next epoch: {self.current_epoch}. Current step: {self.current_step}")


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

    def _save_images_for_fid(self, images_tensor, base_path, num_images_to_save):
        """Saves a batch of image tensors to a directory for FID calculation."""
        os.makedirs(base_path, exist_ok=True)
        images_tensor = denormalize_image(images_tensor) # [-1,1] to [0,1]
        for i in range(min(images_tensor.size(0), num_images_to_save)):
            image = images_tensor[i]
            # Convert to PIL Image and save
            # torchvision.transforms.ToPILImage()(image.cpu()).save(os.path.join(base_path, f"img_{i}.png"))
            # Using vutils.save_image for single image:
            vutils.save_image(image.cpu(), os.path.join(base_path, f"img_{i}.png"), normalize=False)


    def calculate_fid_score(self):
        """Calculates FID score between generated images and real images."""
        if not self.config.enable_fid_calculation or calculate_fid_given_paths is None:
            print("FID calculation skipped (either disabled or pytorch-fid not installed).")
            return float('nan')

        print("Calculating FID score...")
        self.G.eval()

        # Clean up previous FID image directories
        if os.path.exists(self.fid_temp_real_path):
            shutil.rmtree(self.fid_temp_real_path)
        if os.path.exists(self.fid_temp_fake_path):
            shutil.rmtree(self.fid_temp_fake_path)
        os.makedirs(self.fid_temp_real_path, exist_ok=True)
        os.makedirs(self.fid_temp_fake_path, exist_ok=True)

        num_fid_images = self.config.fid_num_images
        fid_batch_size = self.config.fid_batch_size

        # --- Generate and save fake images ---
        print(f"Generating {num_fid_images} fake images for FID...")
        generated_count = 0
        # Use a temporary dataloader for generating fake images, as G needs real image structures
        # (segments, adj) if the GAN architecture requires them (like gan5).
        # We can use the existing self.dataloader but iterate without shuffle.
        temp_dataloader_for_g_inputs = get_dataloader(self.config, shuffle=False)

        pbar_fake = tqdm(total=num_fid_images, desc="Generating Fake FID Images")
        for batch_data in temp_dataloader_for_g_inputs:
            if generated_count >= num_fid_images:
                break

            real_images_batch = batch_data["image"].to(self.device) # Used for structure by G
            segments_map_batch = batch_data["segments"].to(self.device)
            adj_matrix_batch = batch_data["adj"].to(self.device)

            current_gen_batch_size = min(fid_batch_size, real_images_batch.size(0))
            # Ensure we don't generate more than num_fid_images in total
            current_gen_batch_size = min(current_gen_batch_size, num_fid_images - generated_count)
            if current_gen_batch_size <= 0: break


            z = torch.randn(current_gen_batch_size, self.config.z_dim, device=self.device)

            with torch.no_grad():
                fake_images = self.G(z,
                                     real_images_batch[:current_gen_batch_size],
                                     segments_map_batch[:current_gen_batch_size],
                                     adj_matrix_batch[:current_gen_batch_size])

            # Save these fake images
            for i in range(fake_images.size(0)):
                if generated_count < num_fid_images:
                    img_tensor = denormalize_image(fake_images[i].cpu())
                    vutils.save_image(img_tensor, os.path.join(self.fid_temp_fake_path, f"fake_{generated_count}.png"), normalize=False)
                    generated_count += 1
                    pbar_fake.update(1)
                else:
                    break
        pbar_fake.close()
        if generated_count < num_fid_images:
            print(f"Warning: Only generated {generated_count}/{num_fid_images} fake images for FID due to dataset size.")


        # --- Save real images ---
        # TODO: Implement option for config.path_to_real_images_for_fid
        # For now, use images from the current dataset.
        print(f"Saving {num_fid_images} real images for FID...")
        saved_real_count = 0
        real_dataloader_for_fid = get_dataloader(self.config, shuffle=False) # Fresh dataloader

        pbar_real = tqdm(total=num_fid_images, desc="Saving Real FID Images")
        for batch_data in real_dataloader_for_fid:
            if saved_real_count >= num_fid_images:
                break
            real_images_batch = batch_data["image"].to(self.device) # These are already normalized [-1,1]

            for i in range(real_images_batch.size(0)):
                if saved_real_count < num_fid_images:
                    img_tensor = denormalize_image(real_images_batch[i].cpu()) # Denorm to [0,1]
                    vutils.save_image(img_tensor, os.path.join(self.fid_temp_real_path, f"real_{saved_real_count}.png"), normalize=False)
                    saved_real_count += 1
                    pbar_real.update(1)
                else:
                    break
        pbar_real.close()
        if saved_real_count < num_fid_images:
             print(f"Warning: Only saved {saved_real_count}/{num_fid_images} real images for FID due to dataset size.")


        # --- Calculate FID ---
        if generated_count == 0 or saved_real_count == 0:
            print("Not enough images generated/saved for FID calculation. Skipping.")
            self.G.train()
            return float('nan')

        try:
            # FID calculation expects images in range [0, 255], uint8, but pytorch-fid handles [0,1] float PNGs.
            # The images saved by vutils.save_image(tensor, normalize=False) with input tensor in [0,1] are suitable.
            fid_value = calculate_fid_given_paths(
                paths=[self.fid_temp_real_path, self.fid_temp_fake_path],
                batch_size=fid_batch_size, # Batch size for Inception model processing
                device=self.device,
                dims=2048, # Standard InceptionV3 feature dimension for FID
                num_workers=self.config.num_workers
            )
            print(f"FID Score: {fid_value:.4f}")
        except Exception as e:
            print(f"Error calculating FID: {e}")
            fid_value = float('nan')

        # Clean up temporary directories
        # shutil.rmtree(self.fid_temp_real_path)
        # shutil.rmtree(self.fid_temp_fake_path)
        # print("Cleaned up temporary FID image directories.")
        # It might be useful to keep these directories for inspection, so cleanup is commented out.

        self.G.train() # Set generator back to training mode
        return fid_value

    def train(self):
        print("Starting training...")
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()

            if epoch % self.config.sample_freq_epoch == 0:
                self.generate_samples(epoch, self.current_step)

            if self.config.enable_fid_calculation and epoch > 0 and \
               (epoch % self.config.fid_freq_epoch == 0 or epoch == self.config.num_epochs - 1):
                fid_score = self.calculate_fid_score()
                log_to_wandb(self.wandb_run, {"FID_Score": fid_score}, step=self.current_step)


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
