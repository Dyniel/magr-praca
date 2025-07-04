import os
import torch
import dataclasses
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils
import shutil
from PIL import Image
from torchvision import transforms  # Added for _save_images_for_fid

from src.models import (
    Generator as GeneratorGan5, Discriminator as DiscriminatorGan5,
    GraphEncoderGAT, GeneratorCNN, DiscriminatorCNN
)
from src.data_loader import get_dataloader
from src.utils import (
    save_checkpoint, load_checkpoint, setup_wandb, log_to_wandb,
    denormalize_image
)

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    calculate_fid_given_paths = None
    print("Warning: pytorch-fid not found. FID calculation will be disabled. "
          "Install with: pip install pytorch-fid")


class Trainer:
    def __init__(self, config):
        print("DEBUG: Trainer __init__ started.")
        self.config = config
        if calculate_fid_given_paths is None and hasattr(self.config,
                                                         'enable_fid_calculation') and self.config.enable_fid_calculation:
            print("FID calculation has been disabled because pytorch-fid is not installed.")
            self.config.enable_fid_calculation = False

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        os.makedirs(self.config.output_dir_run, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.config.output_dir_run, "checkpoints")
        self.samples_dir = os.path.join(self.config.output_dir_run, "samples")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model_architecture = config.model.architecture
        self.E = None
        self.optE = None

        if self.model_architecture == "gan5_gcn":
            self.G = GeneratorGan5(config).to(self.device)
            self.D = DiscriminatorGan5(config).to(self.device)
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.loss_fn_d = self._loss_d_gan5
            self.loss_fn_g = self._loss_g_gan5
            print("Initialized gan5_gcn models and optimizers.")
        elif self.model_architecture == "gan6_gat_cnn":
            self.E = GraphEncoderGAT(config).to(self.device)
            self.G = GeneratorCNN(config).to(self.device)
            self.D = DiscriminatorCNN(config).to(self.device)
            self.optE = optim.Adam(self.E.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.loss_fn_d = self._loss_d_gan6
            self.loss_fn_g = self._loss_g_gan6
            print("Initialized gan6_gat_cnn models (E, G, D) and optimizers.")
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        self.dataloader = get_dataloader(config, shuffle=True)
        self.fixed_sample_batch = self._prepare_fixed_sample_batch()

        model_to_watch = [self.G,
                          self.E] if self.model_architecture == "gan6_gat_cnn" and self.E is not None else self.G
        self.wandb_run = setup_wandb(config, model_to_watch, project_name=config.wandb_project_name, watch_model=True)

        self.current_epoch = 0
        self.current_step = 0

        self.fid_temp_real_path = os.path.join(self.config.output_dir_run, "fid_real_images_temp")
        self.fid_temp_fake_path = os.path.join(self.config.output_dir_run, "fid_fake_images_temp")

        if hasattr(self.config, 'resume_checkpoint_path') and self.config.resume_checkpoint_path and os.path.exists(
                self.config.resume_checkpoint_path):
            self.load_training_checkpoint(self.config.resume_checkpoint_path)
        else:
            if hasattr(self.config, 'resume_checkpoint_path') and self.config.resume_checkpoint_path:
                print(
                    f"Warning: resume_checkpoint_path '{self.config.resume_checkpoint_path}' not found. Starting from scratch.")
            self.current_epoch = 0
            self.current_step = 0

    def load_training_checkpoint(self, checkpoint_path):
        optimizer_e_to_load = self.optE if self.model_architecture == "gan6_gat_cnn" else None
        model_e_to_load = self.E if self.model_architecture == "gan6_gat_cnn" else None
        start_epoch, current_step = load_checkpoint(
            checkpoint_path,
            self.G, self.D, model_e_to_load,
            self.optG, self.optD, optimizer_e_to_load,
            self.device
        )
        self.current_epoch = start_epoch + 1
        self.current_step = current_step
        print(f"Resumed from checkpoint. Starting next epoch: {self.current_epoch}. Current step: {self.current_step}")

    def _prepare_fixed_sample_batch(self):
        try:
            raw_fixed_batch = next(iter(self.dataloader))
            num_samples = self.config.num_samples_to_log

            if self.model_architecture == "gan5_gcn":
                if not all(k in raw_fixed_batch for k in ["image", "segments", "adj"]):
                    print(
                        f"Warning: Fixed sample batch for gan5_gcn missing expected keys. Batch content: {raw_fixed_batch.keys() if isinstance(raw_fixed_batch, dict) else 'Not a dict'}")
                    return None
                return {
                    "image": raw_fixed_batch["image"][:num_samples].to(self.device),
                    "segments": raw_fixed_batch["segments"][:num_samples].to(self.device),
                    "adj": raw_fixed_batch["adj"][:num_samples].to(self.device)
                }
            elif self.model_architecture == "gan6_gat_cnn":
                if not (isinstance(raw_fixed_batch, tuple) and len(raw_fixed_batch) == 2):
                    print(f"Warning: Fixed sample batch for gan6_gat_cnn not in expected tuple format.")
                    return None
                real_images_tensor, graph_batch_pyg = raw_fixed_batch
                return {
                    "image": real_images_tensor[:num_samples].to(self.device),
                    "graph_batch": graph_batch_pyg[:num_samples].to(self.device)
                }
            return None
        except StopIteration:
            print(
                "Warning: DataLoader exhausted before preparing fixed sample batch. Ensure debug_num_images is >= batch_size.")
            return None
        except Exception as e:
            print(f"Could not prepare fixed sample batch: {e}. Sample generation might be affected.")
            return None

    def _loss_d_gan5(self, d_real_logits, d_fake_logits):
        # Original gan5 D loss: (F.softplus(-r + f) + F.softplus(f - r)).mean()
        # This was (F.softplus(-d_real_logits + d_fake_logits.mean()) + F.softplus(d_fake_logits - d_real_logits.mean())).mean()
        # The implementation was actually per-sample: (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()
        return (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()

    def _loss_g_gan5(self, d_fake_for_g_logits):
        return -d_fake_for_g_logits.mean()

    def _loss_d_gan6(self, d_real_logits, d_fake_logits):
        real_loss = self.bce_loss(d_real_logits, torch.ones_like(d_real_logits))
        fake_loss = self.bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))
        return (real_loss + fake_loss) * 0.5

    def _loss_g_gan6(self, d_fake_for_g_logits):
        return self.bce_loss(d_fake_for_g_logits, torch.ones_like(d_fake_for_g_logits))

    def _r1_gradient_penalty(self, real_images, d_real_logits):
        grad_real = torch.autograd.grad(
            outputs=d_real_logits.sum(), inputs=real_images, create_graph=True,
            allow_unused=True
        )[0]
        if grad_real is None:
            print("Warning: grad_real is None in R1 penalty. Skipping penalty term.")
            return torch.tensor(0.0, device=self.device)
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        return grad_penalty

    def train_epoch(self):
        if self.model_architecture == "gan5_gcn":
            self.G.train()
            self.D.train()
        elif self.model_architecture == "gan6_gat_cnn":
            if self.E: self.E.train()
            self.G.train()
            self.D.train()

        loop = tqdm(self.dataloader, desc=f"Epoch [{self.current_epoch}/{self.config.num_epochs}]")

        # Initialize accumulators for various loss components for logging
        epoch_total_d_loss = 0.0
        epoch_total_g_loss = 0.0
        epoch_total_d_loss_adv = 0.0
        epoch_total_r1_penalty = 0.0
        epoch_d_real_logits_sum = 0.0
        epoch_d_fake_logits_sum = 0.0
        num_optimizer_steps_this_epoch = 0

        # Zero gradients at the beginning of the epoch accumulation cycle
        # Actual zeroing will happen after each optimizer step.
        # For the very first step, ensure optimizers are zeroed.
        self.optD.zero_grad(set_to_none=True)
        self.optG.zero_grad(set_to_none=True)
        if self.optE: self.optE.zero_grad(set_to_none=True)

        # Accumulators for one effective batch (i.e., over gradient_accumulation_steps)
        current_accum_d_loss = 0.0
        current_accum_g_loss = 0.0
        current_accum_d_loss_adv = 0.0
        current_accum_r1_penalty = 0.0
        current_accum_d_real_logits = 0.0
        current_accum_d_fake_logits = 0.0
        micro_batch_count_for_accum = 0

        for batch_idx, raw_batch_data in enumerate(loop):
            # Per micro-batch losses (will be scaled for backward)
            lossD_micro = torch.tensor(0.0, device=self.device)
            lossG_micro = torch.tensor(0.0, device=self.device)
            lossD_adv_micro = torch.tensor(0.0, device=self.device)
            r1_penalty_micro = torch.tensor(0.0, device=self.device)
            d_real_logits_mean_micro = torch.tensor(0.0, device=self.device)
            d_fake_logits_mean_micro = torch.tensor(0.0, device=self.device)
            current_batch_size = 0 # Micro-batch size

            grad_accum_steps = self.config.gradient_accumulation_steps

            # --- Discriminator Training ---
            if self.model_architecture == "gan5_gcn":
                if not (isinstance(raw_batch_data, dict) and all(
                        k in raw_batch_data for k in ["image", "segments", "adj"])):
                    print(f"Warning: Invalid batch data for gan5_gcn: {type(raw_batch_data)}")
                    continue
                real_images = raw_batch_data["image"].to(self.device)
                segments_map = raw_batch_data["segments"].to(self.device)
                adj_matrix = raw_batch_data["adj"].to(self.device)
                current_batch_size = real_images.size(0)
                if current_batch_size == 0: continue

                # The d_updates_per_g_update loop means D is trained more for each micro-batch of G
                # Gradients for D will be accumulated across these inner updates AND across micro-batches
                for _ in range(self.config.d_updates_per_g_update):
                    z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                    with torch.no_grad():
                        fake_images = self.G(z, real_images, segments_map, adj_matrix)

                    d_real_logits = self.D(real_images)
                    d_fake_logits = self.D(fake_images.detach())
                    lossD_adv_iteration = self.loss_fn_d(d_real_logits, d_fake_logits)

                    real_images.requires_grad_(True)
                    d_real_logits_for_gp = self.D(real_images)
                    r1_penalty_iteration = self._r1_gradient_penalty(real_images, d_real_logits_for_gp)
                    real_images.requires_grad_(False)

                    lossD_iteration = lossD_adv_iteration + self.config.r1_gamma * 0.5 * r1_penalty_iteration

                    # Scale loss for gradient accumulation
                    lossD_scaled = lossD_iteration / grad_accum_steps
                    lossD_scaled.backward() # Accumulate gradients for D

                    # Accumulate unscaled losses for logging for this micro-batch's D part
                    lossD_adv_micro += lossD_adv_iteration.item()
                    r1_penalty_micro += r1_penalty_iteration.item()
                    lossD_micro += lossD_iteration.item()

                # Average over d_updates_per_g_update for this micro_batch's D part
                lossD_adv_micro /= self.config.d_updates_per_g_update
                r1_penalty_micro /= self.config.d_updates_per_g_update
                lossD_micro /= self.config.d_updates_per_g_update
                # Logits are from the last D update in the inner loop for this micro_batch
                d_real_logits_mean_micro = d_real_logits.mean()
                d_fake_logits_mean_micro = d_fake_logits.mean()

                # --- Generator Training ---
                z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                fake_images_for_g = self.G(z, real_images, segments_map, adj_matrix)
                d_fake_for_g_logits = self.D(fake_images_for_g)
                lossG_micro_val = self.loss_fn_g(d_fake_for_g_logits)

                lossG_scaled = lossG_micro_val / grad_accum_steps
                lossG_scaled.backward() # Accumulate gradients for G
                lossG_micro = lossG_micro_val.item()


            elif self.model_architecture == "gan6_gat_cnn":
                if not (isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2):
                    print(f"Warning: Invalid batch data for gan6_gat_cnn: {type(raw_batch_data)}")
                    continue
                real_images, graph_batch_pyg = raw_batch_data
                real_images = real_images.to(self.device)
                graph_batch_pyg = graph_batch_pyg.to(self.device)
                current_batch_size = real_images.size(0)
                if current_batch_size == 0: continue

                # --- Discriminator Training ---
                real_images.requires_grad_(True)
                d_real_logits = self.D(real_images)
                with torch.no_grad():
                    z_graph = self.E(graph_batch_pyg)
                    fake_images = self.G(z_graph, current_batch_size)
                d_fake_logits = self.D(fake_images.detach())

                lossD_adv_micro_val = self.loss_fn_d(d_real_logits, d_fake_logits)
                r1_penalty_micro_val = self._r1_gradient_penalty(real_images, d_real_logits)
                real_images.requires_grad_(False)
                lossD_micro_val = lossD_adv_micro_val + self.config.r1_gamma * 0.5 * r1_penalty_micro_val

                lossD_scaled = lossD_micro_val / grad_accum_steps
                lossD_scaled.backward() # Accumulates gradients for D

                lossD_adv_micro = lossD_adv_micro_val.item()
                r1_penalty_micro = r1_penalty_micro_val.item()
                lossD_micro = lossD_micro_val.item()
                d_real_logits_mean_micro = d_real_logits.mean()
                d_fake_logits_mean_micro = d_fake_logits.mean()

                # --- Generator & Encoder Training ---
                z_graph_for_g = self.E(graph_batch_pyg)
                fake_images_for_g = self.G(z_graph_for_g, current_batch_size)
                d_fake_for_g_logits = self.D(fake_images_for_g)
                lossG_and_E_micro_val = self.loss_fn_g(d_fake_for_g_logits)

                lossG_E_scaled = lossG_and_E_micro_val / grad_accum_steps
                lossG_E_scaled.backward() # Accumulates gradients for G and E
                lossG_micro = lossG_and_E_micro_val.item()

            # Accumulate losses for the effective batch
            current_accum_d_loss += lossD_micro
            current_accum_g_loss += lossG_micro
            current_accum_d_loss_adv += lossD_adv_micro
            current_accum_r1_penalty += r1_penalty_micro
            current_accum_d_real_logits += d_real_logits_mean_micro.item()
            current_accum_d_fake_logits += d_fake_logits_mean_micro.item()
            micro_batch_count_for_accum += 1

            # Perform optimizer step if accumulation is complete
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(self.dataloader):
                self.optD.step()
                self.optG.step()
                if self.optE: self.optE.step()

                self.optD.zero_grad(set_to_none=True)
                self.optG.zero_grad(set_to_none=True)
                if self.optE: self.optE.zero_grad(set_to_none=True)

                num_optimizer_steps_this_epoch +=1

                # Log metrics for this effective batch
                avg_d_loss_accum = current_accum_d_loss / micro_batch_count_for_accum
                avg_g_loss_accum = current_accum_g_loss / micro_batch_count_for_accum
                avg_d_loss_adv_accum = current_accum_d_loss_adv / micro_batch_count_for_accum
                avg_r1_penalty_accum = current_accum_r1_penalty / micro_batch_count_for_accum
                avg_d_real_logits_accum = current_accum_d_real_logits / micro_batch_count_for_accum
                avg_d_fake_logits_accum = current_accum_d_fake_logits / micro_batch_count_for_accum

                epoch_total_d_loss += current_accum_d_loss
                epoch_total_g_loss += current_accum_g_loss
                epoch_total_d_loss_adv += current_accum_d_loss_adv
                epoch_total_r1_penalty += current_accum_r1_penalty
                epoch_d_real_logits_sum += current_accum_d_real_logits
                epoch_d_fake_logits_sum += current_accum_d_fake_logits

                if self.current_step % self.config.log_freq_step == 0:
                    log_data = {
                        "Epoch": self.current_epoch,
                        "Step": self.current_step, # This is optimizer step
                        "Loss_D": avg_d_loss_accum,
                        "Loss_D_Adv": avg_d_loss_adv_accum,
                        "R1_Penalty": avg_r1_penalty_accum,
                        "Loss_G": avg_g_loss_accum,
                        "D_Real_Logits_Mean": avg_d_real_logits_accum,
                        "D_Fake_Logits_Mean": avg_d_fake_logits_accum,
                    }
                    log_to_wandb(self.wandb_run, log_data, step=self.current_step)
                    loop.set_postfix(log_data)

                self.current_step += 1 # Increment optimizer step counter

                # Reset accumulators for the next effective batch
                current_accum_d_loss = 0.0
                current_accum_g_loss = 0.0
                current_accum_d_loss_adv = 0.0
                current_accum_r1_penalty = 0.0
                current_accum_d_real_logits = 0.0
                current_accum_d_fake_logits = 0.0
                micro_batch_count_for_accum = 0

        # Calculate and log epoch averages based on optimizer steps
        if num_optimizer_steps_this_epoch > 0 :
            avg_epoch_d_loss = epoch_total_d_loss / (num_optimizer_steps_this_epoch * grad_accum_steps)
            avg_epoch_g_loss = epoch_total_g_loss / (num_optimizer_steps_this_epoch * grad_accum_steps)
            # Note: The above averages might be slightly off if the last batch wasn't full.
            # A more precise way is to average the per-step logged averages, or sum items from all micro_batches.
            # For simplicity, using total items / total micro_batches processed in epoch.
            total_micro_batches_in_epoch = len(self.dataloader)
            avg_epoch_d_loss = epoch_total_d_loss / total_micro_batches_in_epoch if total_micro_batches_in_epoch > 0 else 0
            avg_epoch_g_loss = epoch_total_g_loss / total_micro_batches_in_epoch if total_micro_batches_in_epoch > 0 else 0

            print(f"Epoch {self.current_epoch} finished. Avg D Loss: {avg_epoch_d_loss:.4f}, Avg G Loss: {avg_epoch_g_loss:.4f}")
            log_to_wandb(self.wandb_run,
                        {"Epoch_Avg_D_Loss": avg_epoch_d_loss,
                         "Epoch_Avg_G_Loss": avg_epoch_g_loss,
                         "Epoch": self.current_epoch},
                        step=self.current_step) # Log against the last optimizer step of the epoch
        else:
            print(f"Epoch {self.current_epoch} finished. No optimizer steps taken (dataloader might be empty or too short for accumulation).")


    def generate_samples(self, epoch, step):
        if not self.fixed_sample_batch:
            print("Fixed sample batch not available. Skipping sample generation.")
            return

        if self.model_architecture == "gan5_gcn":
            self.G.eval()
        elif self.model_architecture == "gan6_gat_cnn":
            if self.E: self.E.eval()
            self.G.eval()

        with torch.no_grad():
            generated_samples = None
            real_samples_for_log = None
            num_fixed_samples = 0
            if self.fixed_sample_batch and "image" in self.fixed_sample_batch:
                num_fixed_samples = self.fixed_sample_batch["image"].size(0)
            else:  # Should not happen if _prepare_fixed_sample_batch worked
                print("Warning: fixed_sample_batch is not as expected in generate_samples.")
                return

            if self.model_architecture == "gan5_gcn":
                z_sample = torch.randn(num_fixed_samples, self.config.model.z_dim, device=self.device)
                generated_samples = self.G(
                    z_sample,
                    self.fixed_sample_batch["image"],
                    self.fixed_sample_batch["segments"],
                    self.fixed_sample_batch["adj"]
                )
                real_samples_for_log = self.fixed_sample_batch["image"]

            elif self.model_architecture == "gan6_gat_cnn":
                if "graph_batch" not in self.fixed_sample_batch or self.E is None:
                    print("Warning: graph_batch or Encoder not available for gan6 sample generation.")
                    return
                z_graph = self.E(self.fixed_sample_batch["graph_batch"])
                generated_samples = self.G(z_graph, num_fixed_samples)
                real_samples_for_log = self.fixed_sample_batch["image"]

            if generated_samples is not None and real_samples_for_log is not None:
                generated_samples = denormalize_image(generated_samples)
                real_samples_for_log = denormalize_image(real_samples_for_log)
                nrow_val = max(1, int(np.sqrt(generated_samples.size(0)))) if generated_samples.size(0) > 0 else 1

                img_grid_fake = vutils.make_grid(generated_samples, normalize=False, nrow=nrow_val)
                img_grid_real = vutils.make_grid(real_samples_for_log, normalize=False, nrow=nrow_val)

                vutils.save_image(img_grid_fake,
                                  os.path.join(self.samples_dir, f"fake_samples_epoch_{epoch:04d}_step_{step}.png"))
                if epoch == 0 and self.current_step <= self.config.log_freq_step * 2:
                    vutils.save_image(img_grid_real,
                                      os.path.join(self.samples_dir, f"real_samples_epoch_{epoch:04d}.png"))

                if self.wandb_run:
                    log_data_img = {f"Generated_Samples_Epoch_{epoch}": self.wandb_run.Image(img_grid_fake)}
                    if epoch == 0 and self.current_step <= self.config.log_freq_step * 2:
                        log_data_img["Real_Samples"] = self.wandb_run.Image(img_grid_real)
                    log_to_wandb(self.wandb_run, log_data_img, step=step)
            else:
                print(
                    "Warning: Could not generate samples for logging due to missing batch data or generated_samples being None.")

        if self.model_architecture == "gan5_gcn":
            self.G.train()
        elif self.model_architecture == "gan6_gat_cnn":
            if self.E: self.E.train()
            self.G.train()

    def _save_images_for_fid(self, images_tensor, base_path, start_idx, num_to_save_this_batch):
        os.makedirs(base_path, exist_ok=True)
        images_tensor_denorm = denormalize_image(images_tensor)
        for i in range(num_to_save_this_batch):
            # Convert to PIL Image and save, ensuring correct format
            # ToPILImage expects CxHxW, values in [0,1]
            img_to_save = transforms.ToPILImage()(images_tensor_denorm[i].cpu())
            img_to_save.save(os.path.join(base_path, f"img_{start_idx + i}.png"))

    def calculate_fid_score(self):
        if not hasattr(self.config,
                       'enable_fid_calculation') or not self.config.enable_fid_calculation or calculate_fid_given_paths is None:
            return float('nan')

        print("Calculating FID score...")
        original_modes = {}
        if self.model_architecture == "gan5_gcn":
            original_modes['G'] = self.G.training
            self.G.eval()
        elif self.model_architecture == "gan6_gat_cnn":
            if self.E: original_modes['E'] = self.E.training; self.E.eval()
            original_modes['G'] = self.G.training
            self.G.eval()

        if os.path.exists(self.fid_temp_real_path):
            shutil.rmtree(self.fid_temp_real_path)
        if os.path.exists(self.fid_temp_fake_path):
            shutil.rmtree(self.fid_temp_fake_path)
        os.makedirs(self.fid_temp_real_path, exist_ok=True)
        os.makedirs(self.fid_temp_fake_path, exist_ok=True)

        num_fid_images = self.config.fid_num_images
        fid_batch_size_gen = self.config.fid_batch_size

        generated_count = 0
        # Create a new dataloader instance for FID to ensure it starts from the beginning
        temp_dataloader_for_fid = get_dataloader(self.config, shuffle=False)

        pbar_fake = tqdm(total=num_fid_images, desc="Generating Fake FID Images")
        for raw_batch_data_fid in temp_dataloader_for_fid:
            if generated_count >= num_fid_images: break

            actual_batch_size_this_iter = 0
            fake_images_batch = None

            if self.model_architecture == "gan5_gcn":
                real_images_fid = raw_batch_data_fid["image"].to(self.device)
                segments_map_fid = raw_batch_data_fid["segments"].to(self.device)
                adj_matrix_fid = raw_batch_data_fid["adj"].to(self.device)
                actual_batch_size_this_iter = real_images_fid.size(0)

                current_gen_count = min(fid_batch_size_gen, actual_batch_size_this_iter,
                                        num_fid_images - generated_count)
                if current_gen_count <= 0: continue
                z = torch.randn(current_gen_count, self.config.model.z_dim, device=self.device)
                with torch.no_grad():
                    fake_images_batch = self.G(z,
                                               real_images_fid[:current_gen_count],
                                               segments_map_fid[:current_gen_count],
                                               adj_matrix_fid[:current_gen_count])

            elif self.model_architecture == "gan6_gat_cnn":
                if self.E is None: continue  # Should not happen if initialized correctly
                real_images_fid, graph_batch_pyg_fid = raw_batch_data_fid
                graph_batch_pyg_fid = graph_batch_pyg_fid.to(self.device)
                actual_batch_size_this_iter = graph_batch_pyg_fid.num_graphs if hasattr(graph_batch_pyg_fid,
                                                                                        'num_graphs') else (
                    graph_batch_pyg_fid.ptr.numel() - 1 if hasattr(graph_batch_pyg_fid,
                                                                   'ptr') else real_images_fid.size(0))

                current_gen_count = min(fid_batch_size_gen, actual_batch_size_this_iter,
                                        num_fid_images - generated_count)
                if current_gen_count <= 0: continue

                with torch.no_grad():
                    # Slice the graph batch if current_gen_count is less than the number of graphs in batch
                    sliced_graph_batch = graph_batch_pyg_fid[:current_gen_count]
                    z_graph = self.E(sliced_graph_batch)
                    fake_images_batch = self.G(z_graph, current_gen_count)

            if fake_images_batch is None: continue
            self._save_images_for_fid(fake_images_batch, self.fid_temp_fake_path, generated_count,
                                      fake_images_batch.size(0))
            generated_count += fake_images_batch.size(0)
            pbar_fake.update(fake_images_batch.size(0))
        pbar_fake.close()

        saved_real_count = 0
        real_dataloader_for_fid_save = get_dataloader(self.config, shuffle=False)
        pbar_real = tqdm(total=num_fid_images, desc="Saving Real FID Images")
        for raw_batch_data_fid_real in real_dataloader_for_fid_save:
            if saved_real_count >= num_fid_images: break

            if self.model_architecture == "gan6_gat_cnn":
                real_images_batch_save, _ = raw_batch_data_fid_real
            else:
                real_images_batch_save = raw_batch_data_fid_real["image"]

            real_images_batch_save = real_images_batch_save.to(self.device)
            num_to_save_this_batch = min(real_images_batch_save.size(0), num_fid_images - saved_real_count)
            if num_to_save_this_batch <= 0: continue

            self._save_images_for_fid(real_images_batch_save[:num_to_save_this_batch], self.fid_temp_real_path,
                                      saved_real_count, num_to_save_this_batch)
            saved_real_count += num_to_save_this_batch
            pbar_real.update(num_to_save_this_batch)
        pbar_real.close()

        fid_value = float('nan')
        # Ensure enough images were actually saved before attempting FID calculation
        if generated_count >= min(num_fid_images, 10) and saved_real_count >= min(num_fid_images,
                                                                                  10):  # Min 10 images to avoid error with FID tool
            try:
                fid_value = calculate_fid_given_paths(
                    paths=[self.fid_temp_real_path, self.fid_temp_fake_path],
                    batch_size=self.config.fid_batch_size,
                    device=self.device,
                    dims=2048,
                    num_workers=self.config.num_workers
                )
                print(f"FID Score: {fid_value:.4f}")
            except Exception as e:
                print(f"Error calculating FID: {e}")
        else:
            print(
                f"Skipping FID: Not enough images. Needed at least 10 of each. Got Real: {saved_real_count}, Got Fake: {generated_count}")

        if self.model_architecture == "gan5_gcn":
            if 'G' in original_modes: self.G.train(original_modes['G'])
        elif self.model_architecture == "gan6_gat_cnn":
            if self.E and 'E' in original_modes: self.E.train(original_modes['E'])
            if 'G' in original_modes: self.G.train(original_modes['G'])
        return fid_value

    def train(self):
        print("Starting training...")
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()

            if epoch % self.config.sample_freq_epoch == 0:
                self.generate_samples(epoch, self.current_step)

            if hasattr(self.config, 'enable_fid_calculation') and self.config.enable_fid_calculation and \
                    (epoch % self.config.fid_freq_epoch == 0 or epoch == self.config.num_epochs - 1):
                fid_score = self.calculate_fid_score()
                if not np.isnan(fid_score):
                    log_to_wandb(self.wandb_run, {"FID_Score": fid_score, "Epoch": self.current_epoch},
                                 step=self.current_step)

            if epoch % self.config.checkpoint_freq_epoch == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_data = {
                    'epoch': epoch,
                    'step': self.current_step,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                    'optG_state_dict': self.optG.state_dict(),
                    'optD_state_dict': self.optD.state_dict(),
                    'config': dataclasses.asdict(self.config)  # Save config as dict
                }
                if self.model_architecture == "gan6_gat_cnn" and self.E is not None and self.optE is not None:
                    checkpoint_data['E_state_dict'] = self.E.state_dict()
                    checkpoint_data['optE_state_dict'] = self.optE.state_dict()

                save_checkpoint(checkpoint_data, is_best=False,
                                filename=os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch:04d}.pth.tar"))
                print(
                    f"Checkpoint saved to {os.path.join(self.checkpoints_dir, f'checkpoint_epoch_{epoch:04d}.pth.tar')}")

        print("Training finished.")
        if self.wandb_run:
            self.wandb_run.finish()


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf
    from configs.base_config import BaseConfig # Assuming BaseConfig is in configs.base_config

    def main():
        # --- Argument Parsing and Configuration Loading ---
        parser = argparse.ArgumentParser(description="Train a GAN model.")
        parser.add_argument("--config_file", type=str, default=None,
                            help="Path to a YAML configuration file to override base defaults.")
        # Allow unknown args for OmegaConf to parse as dot-list overrides
        args, unknown_args = parser.parse_known_args()

        # Start with the structured default config
        conf = OmegaConf.structured(BaseConfig)

        # Load config from YAML file if provided
        if args.config_file:
            try:
                file_conf = OmegaConf.load(args.config_file)
                conf = OmegaConf.merge(conf, file_conf)
                print(f"Loaded configuration from {args.config_file}")
            except FileNotFoundError:
                print(f"Warning: Config file {args.config_file} not found. Using defaults and CLI overrides.")
            except Exception as e:
                print(f"Error loading config file {args.config_file}: {e}. Using defaults and CLI overrides.")


        # Apply command-line overrides (e.g., batch_size=2 num_epochs=3)
        # These need to be in the format: param.subparam=value or param=value
        cli_conf_list = []
        temp_dict = {}
        for i in range(0, len(unknown_args), 2 if '=' in ''.join(unknown_args) else 1):
            arg = unknown_args[i]
            if '=' in arg: # handles param=value
                cli_conf_list.append(arg)
            elif i + 1 < len(unknown_args): # handles param value
                cli_conf_list.append(f"{arg}={unknown_args[i+1]}")
                # This part is tricky with current OmegaConf parsing from list of strings if not "key=value"
                # For simplicity, we'll assume users will pass "key=value" for CLI overrides
                # A more robust way would be to parse them into a dict first, then OmegaConf.from_dotlist or OmegaConf.from_dict
            else:
                print(f"Warning: Ignoring orphaned CLI argument: {arg}")

        # Re-parse unknown_args assuming they are OmegaConf dot-list style (e.g., batch_size=2)
        # The previous loop was a bit convoluted. OmegaConf can handle a list of "key=value" strings.
        # Let's simplify the parsing of unknown_args for OmegaConf.
        # We expect overrides like 'batch_size=2' 'num_epochs=3'

        # Filter out any non-key-value pair arguments from unknown_args, like flags without values if any
        # For example, if someone runs `python train.py batch_size=2 --some_flag num_epochs=3`
        # OmegaConf expects a list of strings like ["batch_size=2", "num_epochs=3"]

        # Let's refine the CLI override parsing.
        # The original command was: python -m scripts.train --config_file configs/experiment_config.yaml batch_size=2 num_epochs=3 ...
        # `unknown_args` would be ['batch_size=2', 'num_epochs=3', 'debug_num_images=4', 'use_wandb=False', 'num_workers=0']
        # This format is directly consumable by OmegaConf.from_cli() or by merging with OmegaConf.from_dotlist()

        if unknown_args:
            try:
                # OmegaConf.from_cli() is designed for sys.argv directly.
                # For already parsed unknown_args (list of strings), OmegaConf.from_dotlist is more appropriate.
                # However, OmegaConf.from_dotlist expects dot-separated paths for nested keys.
                # The provided CLI arguments are simple key=value for top-level or direct model keys.
                # OmegaConf.merge accepts multiple dicts or OmegaConf objects.
                # We can create a new OmegaConf object from these CLI args.

                # Let's try to create a dotlist from the unknown_args.
                # Example: ['batch_size=2', 'model.z_dim=128']
                # This is what OmegaConf.from_dotlist expects.
                # The arguments `batch_size=2` are already in this format.

                cli_overrides = OmegaConf.from_dotlist(unknown_args)
                conf = OmegaConf.merge(conf, cli_overrides)
                print(f"Applied CLI overrides: {unknown_args}")
            except Exception as e:
                print(f"Error applying CLI overrides: {e}")
                parser.print_help()
                return


        # --- Trainer Initialization and Training ---
        # Ensure the output directory defined in the config is created
        # The Trainer class __init__ already does this with config.output_dir_run
        # os.makedirs(conf.output_dir_run, exist_ok=True) # Not strictly needed here due to Trainer

        print("Final configuration after merges:")
        print(OmegaConf.to_yaml(conf))

        try:
            # Convert the OmegaConf object to an instance of the BaseConfig dataclass
            # This ensures that __post_init__ is called.
            conf = OmegaConf.to_object(conf)

            trainer = Trainer(config=conf)
            print("Trainer initialized. Starting training...")
            trainer.train()
        except Exception as e:
            print(f"An error occurred during trainer initialization or training: {e}")
            import traceback
            traceback.print_exc()


    main()