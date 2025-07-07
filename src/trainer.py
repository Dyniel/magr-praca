import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils  # For making image grids
from torchvision import transforms  # Added for ToPILImage

# Local imports
from src.models import (
    Generator as GeneratorGan5, Discriminator as DiscriminatorGan5,  # gan5 models
    GraphEncoderGAT, GeneratorCNN, DiscriminatorCNN  # gan6 models
)
from src.data_loader import get_dataloader
from torch_geometric.data import Batch as PyGBatch  # For collating lists of Data objects
from torch_geometric.data import Data as PyGData  # For type checking
from src.utils import (
    save_checkpoint, load_checkpoint, setup_wandb, log_to_wandb,
    denormalize_image  # To convert [-1,1] to [0,1] for logging images
)
import shutil  # For FID image directory handling
from PIL import Image  # For saving images for FID

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
            self.config.enable_fid_calculation = False  # Disable if library not found
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
            torch.backends.cudnn.deterministic = True  # Can slow down, but good for reproducibility
            torch.backends.cudnn.benchmark = False  # Disable benchmark mode for determinism

        # Initialize Models, Optimizers, and Loss based on architecture
        self.model_architecture = config.model.architecture
        self.E = None  # For gan6
        self.optE = None  # For gan6

        if self.model_architecture == "gan5_gcn":
            self.G = GeneratorGan5(config).to(self.device)
            self.D = DiscriminatorGan5(config).to(self.device)
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.loss_fn_g = self._loss_g_gan5
            self.loss_fn_d = self._loss_d_gan5
            print("Initialized gan5_gcn models and optimizers.")
        elif self.model_architecture == "gan6_gat_cnn":
            self.E = GraphEncoderGAT(config).to(self.device)
            self.G = GeneratorCNN(config).to(self.device)
            self.D = DiscriminatorCNN(config).to(self.device)
            # Using config.g_lr for both E and G, config.d_lr for D as per legacy/gan6
            self.optE = optim.Adam(self.E.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optG = optim.Adam(self.G.parameters(), lr=config.g_lr, betas=(config.beta1, config.beta2))
            self.optD = optim.Adam(self.D.parameters(), lr=config.d_lr, betas=(config.beta1, config.beta2))
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.loss_fn_g = self._loss_g_gan6
            self.loss_fn_d = self._loss_d_gan6
            print("Initialized gan6_gat_cnn models (E, G, D) and optimizers.")
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        # Initialize DataLoader (now depends on model.architecture from config)
        self.dataloader = get_dataloader(config, shuffle=True)

        # A fixed batch for generating samples during training
        self.fixed_sample_batch = self._prepare_fixed_sample_batch()

        # Initialize Weights & Biases (if enabled)
        # Watch G for gan5, or G and E for gan6 (D is usually not watched)
        model_to_watch = [self.G, self.E] if self.model_architecture == "gan6_gat_cnn" else self.G
        self.wandb_run = setup_wandb(config, model_to_watch, project_name=config.wandb_project_name, watch_model=True)

        self.current_epoch = 0
        self.current_step = 0

        self.fid_temp_real_path = os.path.join(self.config.output_dir_run, "fid_real_images_temp")
        self.fid_temp_fake_path = os.path.join(self.config.output_dir_run, "fid_fake_images_temp")

        if self.config.resume_checkpoint_path and os.path.exists(self.config.resume_checkpoint_path):
            self.load_training_checkpoint(self.config.resume_checkpoint_path)
        else:
            if self.config.resume_checkpoint_path:  # Path given but not found
                print(
                    f"Warning: resume_checkpoint_path '{self.config.resume_checkpoint_path}' not found. Starting from scratch.")
            self.current_epoch = 0
            self.current_step = 0

    def load_training_checkpoint(self, checkpoint_path):
        """Loads G, D, (E), optimizers, epoch, and step from a checkpoint."""
        optE_to_load = self.optE if self.model_architecture == "gan6_gat_cnn" else None
        start_epoch, current_step = load_checkpoint(
            checkpoint_path,
            self.G, self.D, self.E,  # Pass E, load_checkpoint will handle if it's None
            self.optG, self.optD, optE_to_load,
            self.device
        )
        self.current_epoch = start_epoch + 1
        self.current_step = current_step
        print(f"Resumed from checkpoint. Starting next epoch: {self.current_epoch}. Current step: {self.current_step}")

    def _prepare_fixed_sample_batch(self):
        """Prepares a fixed batch of data for consistent sample generation."""
        try:
            # Dataloader yields different structures based on architecture
            raw_fixed_batch = next(iter(self.dataloader))
            num_samples = self.config.num_samples_to_log

            if self.model_architecture == "gan5_gcn":
                return {
                    "image": raw_fixed_batch["image"][:num_samples].to(self.device),
                    "segments": raw_fixed_batch["segments"][:num_samples].to(self.device),
                    "adj": raw_fixed_batch["adj"][:num_samples].to(self.device)
                }
            elif self.model_architecture == "gan6_gat_cnn":
                # For gan6, batch is (real_images_tensor, graph_batch_pyg)
                real_images_tensor, graph_batch_pyg = raw_fixed_batch
                # We need to select a subset of graphs from graph_batch_pyg if PyGBatch supports it,
                # or just use the whole batch if it's already small enough.
                # For simplicity, if num_samples < batch_size, we might use the first num_samples from tensors
                # and potentially re-batch the graphs or ensure fixed_batch_size for samples is handled.
                # Let's assume the dataloader's batch for fixed samples is small enough or use all of it.
                # The GraphEncoder expects a PyGBatch.
                # We need to slice the graph_batch_pyg. Slicing PyGBatch: batch[:num_samples]

                # Ensure graph_batch_pyg is a PyGBatch object
                if isinstance(graph_batch_pyg, list) and all(isinstance(g, PyGData) for g in graph_batch_pyg):
                    graph_batch_pyg = PyGBatch.from_data_list(graph_batch_pyg)

                # Now, it should be safe to slice and send to device
                sliced_graph_batch = graph_batch_pyg[:num_samples]

                return {
                    "image": real_images_tensor[:num_samples].to(self.device),  # Real images for D
                    "graph_batch": sliced_graph_batch.to(self.device)  # Graph data for E
                }
            return None
        except Exception as e:
            print(f"Could not prepare fixed sample batch: {e}. Sample generation might be affected.")
            return None

    # --- Loss functions for gan5_gcn ---
    def _loss_d_gan5(self, d_real_logits, d_fake_logits):
        return (F.softplus(-d_real_logits + d_fake_logits) + F.softplus(d_fake_logits - d_real_logits)).mean()

    def _loss_g_gan5(self, d_fake_for_g_logits):
        return -d_fake_for_g_logits.mean()

    # --- Loss functions for gan6_gat_cnn ---
    def _loss_d_gan6(self, d_real_logits, d_fake_logits):
        real_loss = self.bce_loss(d_real_logits, torch.ones_like(d_real_logits))
        fake_loss = self.bce_loss(d_fake_logits, torch.zeros_like(d_fake_logits))
        return (real_loss + fake_loss) * 0.5

    def _loss_g_gan6(self, d_fake_for_g_logits):
        return self.bce_loss(d_fake_for_g_logits, torch.ones_like(d_fake_for_g_logits))

    def _r1_gradient_penalty(self, real_images, d_real_logits):
        """Calculates R1 gradient penalty. Common for both architectures."""
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
        total_e_loss = 0.0  # For gan6

        for batch_idx, raw_batch_data in enumerate(loop):
            current_batch_size = 0
            lossD = torch.tensor(0.0)
            lossG = torch.tensor(0.0)
            lossE = torch.tensor(0.0)  # For gan6
            lossD_adv = torch.tensor(0.0)
            r1_penalty = torch.tensor(0.0)
            d_real_logits_mean = torch.tensor(0.0)
            d_fake_logits_mean = torch.tensor(0.0)

            if self.model_architecture == "gan5_gcn":
                real_images = raw_batch_data["image"].to(self.device)
                segments_map = raw_batch_data["segments"].to(self.device)
                adj_matrix = raw_batch_data["adj"].to(self.device)
                current_batch_size = real_images.size(0)

                # --- Train Discriminator (gan5) ---
                for _ in range(self.config.d_updates_per_g_update):
                    self.optD.zero_grad()
                    z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                    with torch.no_grad():
                        fake_images = self.G(z, real_images, segments_map, adj_matrix)

                    d_real_logits = self.D(real_images)
                    d_fake_logits = self.D(fake_images.detach())  # Use detached fakes

                    lossD_adv = self.loss_fn_d(d_real_logits, d_fake_logits)

                    real_images.requires_grad_(True)
                    d_real_logits_for_gp = self.D(real_images)
                    r1_penalty = self._r1_gradient_penalty(real_images, d_real_logits_for_gp)
                    real_images.requires_grad_(False)

                    lossD = lossD_adv + self.config.r1_gamma * 0.5 * r1_penalty
                    lossD.backward()
                    self.optD.step()

                d_real_logits_mean = d_real_logits.mean()  # For logging
                d_fake_logits_mean = d_fake_logits.mean()  # For logging

                # --- Train Generator (gan5) ---
                self.optG.zero_grad()
                z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                fake_images_for_g = self.G(z, real_images, segments_map, adj_matrix)
                d_fake_for_g_logits = self.D(fake_images_for_g)
                lossG = self.loss_fn_g(d_fake_for_g_logits)
                lossG.backward()
                self.optG.step()

            elif self.model_architecture == "gan6_gat_cnn":
                real_images, graph_batch_pyg = raw_batch_data
                real_images = real_images.to(self.device)
                graph_batch_pyg = graph_batch_pyg.to(self.device)
                current_batch_size = real_images.size(0)

                # --- Train Discriminator (gan6) ---
                self.optD.zero_grad()
                real_images.requires_grad_(True)  # For R1 penalty

                d_real_logits = self.D(real_images)

                with torch.no_grad():
                    z_graph = self.E(graph_batch_pyg)
                    fake_images = self.G(z_graph, current_batch_size)

                d_fake_logits = self.D(fake_images.detach())

                lossD_adv = self.loss_fn_d(d_real_logits, d_fake_logits)
                r1_penalty = self._r1_gradient_penalty(real_images, d_real_logits)  # Use d_real_logits directly
                real_images.requires_grad_(False)

                lossD = lossD_adv + self.config.r1_gamma * 0.5 * r1_penalty
                lossD.backward()
                self.optD.step()

                d_real_logits_mean = d_real_logits.mean()  # For logging
                d_fake_logits_mean = d_fake_logits.mean()  # For logging

                # --- Train Generator & Encoder (gan6) ---
                self.optE.zero_grad()
                self.optG.zero_grad()

                z_graph_for_g = self.E(graph_batch_pyg)
                fake_images_for_g = self.G(z_graph_for_g, current_batch_size)
                d_fake_for_g_logits = self.D(fake_images_for_g)

                lossG_and_E = self.loss_fn_g(d_fake_for_g_logits)  # Combined loss for G and E
                lossG_and_E.backward()
                self.optG.step()
                self.optE.step()
                lossG = lossG_and_E  # For logging, treat combined loss as G_loss
                # lossE is implicitly part of lossG here

            total_d_loss += lossD.item()
            total_g_loss += lossG.item()
            if self.model_architecture == "gan6_gat_cnn":
                # For gan6, E is optimized with G. We log the combined G_loss.
                # If we wanted separate E loss, it would need a different formulation.
                pass

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

            self.current_step += 1

        avg_d_loss = total_d_loss / len(self.dataloader)
        avg_g_loss = total_g_loss / len(self.dataloader)
        print(f"Epoch {self.current_epoch} finished. Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")
        log_to_wandb(self.wandb_run, {"Epoch_Avg_D_Loss": avg_d_loss, "Epoch_Avg_G_Loss": avg_g_loss},
                     step=self.current_step)

    def generate_samples(self, epoch, step):
        if not self.fixed_sample_batch:
            print("Fixed sample batch not available. Skipping sample generation.")
            return

        self.G.eval()
        if self.E: self.E.eval()

        with torch.no_grad():
            generated_samples = None
            real_samples_for_log = None
            num_fixed_samples = self.config.num_samples_to_log  # Should match _prepare_fixed_sample_batch

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
                # Ensure fixed_sample_batch["graph_batch"] is correctly sized for num_fixed_samples
                # The graph_batch might contain more graphs than num_samples_to_log if batch_size > num_samples_to_log
                # However, _prepare_fixed_sample_batch already slices graph_batch correctly.
                z_graph = self.E(self.fixed_sample_batch["graph_batch"])
                # GeneratorCNN takes batch_size for z_noise, which should be num_fixed_samples
                generated_samples = self.G(z_graph, num_fixed_samples)
                real_samples_for_log = self.fixed_sample_batch["image"]

            if generated_samples is not None and real_samples_for_log is not None:
                generated_samples = denormalize_image(generated_samples)
                real_samples_for_log = denormalize_image(real_samples_for_log)  # Should be B, C, H, W
                generated_samples = denormalize_image(generated_samples)  # Should be B, C, H, W

                # Save grid of fakes (optional, good for quick view)
                img_grid_fake = vutils.make_grid(generated_samples, normalize=False,
                                                 nrow=int(np.sqrt(generated_samples.size(0))))
                vutils.save_image(img_grid_fake, os.path.join(self.samples_dir,
                                                              f"fake_samples_grid_epoch_{epoch:04d}_step_{step}.png"))

                if self.wandb_run:
                    # Create a W&B Table for comparing real and fake images side-by-side
                    image_comparison_table = self.wandb_run.Table(
                        columns=["Epoch", "Step", "ID", "Real Image", "Generated Image"])

                    num_images_to_log_in_table = generated_samples.size(0)  # Should match num_fixed_samples

                    for i in range(num_images_to_log_in_table):
                        real_img_pil = transforms.ToPILImage()(real_samples_for_log[i].cpu())
                        fake_img_pil = transforms.ToPILImage()(generated_samples[i].cpu())

                        image_comparison_table.add_data(
                            epoch,
                            step,
                            i,  # Sample ID within the batch
                            self.wandb_run.Image(real_img_pil),
                            self.wandb_run.Image(fake_img_pil)
                        )

                    # Log the table
                    log_to_wandb(self.wandb_run, {f"Image_Comparison_Slider_Epoch_{epoch}": image_comparison_table},
                                 step=step)

                    # Optionally, still log the grid of fakes and initial reals if desired
                    # log_images_wandb = {f"Generated_Samples_Grid_Epoch_{epoch}": self.wandb_run.Image(img_grid_fake)}
                    # if epoch == 0 and step <= self.config.log_freq_step * 2:
                    #     img_grid_real = vutils.make_grid(real_samples_for_log, normalize=False, nrow=int(np.sqrt(real_samples_for_log.size(0))))
                    #     log_images_wandb["Real_Samples_Grid"] = self.wandb_run.Image(img_grid_real)
                    #     vutils.save_image(img_grid_real, os.path.join(self.samples_dir, f"real_samples_grid_epoch_{epoch:04d}.png"))
                    # log_to_wandb(self.wandb_run, log_images_wandb, step=step)
            else:
                print(
                    "Warning: Could not generate samples for logging (generated_samples or real_samples_for_log is None).")

        self.G.train()
        if self.E: self.E.train()

    def _evaluate_on_split(self, data_split: str):
        """
        Evaluates the model on a given data split (e.g., "val" or "test").
        Calculates losses and other relevant metrics.
        """
        print(f"Evaluating on {data_split} split...")
        # Ensure drop_last is False for evaluation to use all samples
        eval_dataloader = get_dataloader(self.config, data_split=data_split, shuffle=False, drop_last=False)

        if eval_dataloader is None:
            print(f"No dataloader for {data_split} split (path likely not configured). Skipping evaluation.")
            return {}

        self.G.eval()
        self.D.eval()
        if self.E:
            self.E.eval()

        total_d_loss = 0.0
        total_g_loss = 0.0
        total_d_loss_adv = 0.0
        total_r1_penalty = 0.0
        total_d_real_logits = 0.0
        total_d_fake_logits = 0.0
        num_batches = 0

        with torch.no_grad():
            for raw_batch_data in tqdm(eval_dataloader, desc=f"Evaluating {data_split}"):
                lossD_batch = torch.tensor(0.0, device=self.device)
                lossG_batch = torch.tensor(0.0, device=self.device)
                lossD_adv_batch = torch.tensor(0.0, device=self.device)
                r1_penalty_val = torch.tensor(0.0, device=self.device)  # Renamed to avoid clash
                d_real_logits_mean_batch = torch.tensor(0.0, device=self.device)
                d_fake_logits_mean_batch = torch.tensor(0.0, device=self.device)
                current_batch_size = 0

                if self.model_architecture == "gan5_gcn":
                    if not (isinstance(raw_batch_data, dict) and all(
                            k in raw_batch_data for k in ["image", "segments", "adj"])):
                        print(
                            f"Warning: Invalid batch data for gan5_gcn during {data_split} eval: {type(raw_batch_data)}")
                        continue
                    real_images = raw_batch_data["image"].to(self.device)
                    segments_map = raw_batch_data["segments"].to(self.device)
                    adj_matrix = raw_batch_data["adj"].to(self.device)
                    current_batch_size = real_images.size(0)
                    if current_batch_size == 0: continue

                    z = torch.randn(current_batch_size, self.config.model.z_dim, device=self.device)
                    fake_images = self.G(z, real_images, segments_map, adj_matrix)

                    d_real_logits = self.D(real_images)
                    d_fake_logits = self.D(fake_images)

                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)

                    # R1 penalty calculation during eval:
                    # Requires enabling grad on real_images if it was off.
                    # For simplicity in eval, often only adversarial loss is reported or R1 is optional.
                    # If we want to calculate it:
                    # real_images_for_gp = real_images.clone().requires_grad_(True)
                    # d_real_logits_for_gp = self.D(real_images_for_gp)
                    # r1_penalty_val = self._r1_gradient_penalty(real_images_for_gp, d_real_logits_for_gp)
                    # For now, keep r1_penalty_val as 0 for val/test to avoid complexity unless specified.
                    lossD_batch = lossD_adv_batch  # + self.config.r1_gamma * 0.5 * r1_penalty_val

                    d_fake_for_g_logits = self.D(fake_images)
                    lossG_batch = self.loss_fn_g(d_fake_for_g_logits)
                    d_real_logits_mean_batch = d_real_logits.mean()
                    d_fake_logits_mean_batch = d_fake_logits.mean()

                elif self.model_architecture == "gan6_gat_cnn":
                    if not (isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2):
                        print(
                            f"Warning: Invalid batch data for gan6_gat_cnn during {data_split} eval: {type(raw_batch_data)}")
                        continue
                    real_images, graph_batch_pyg = raw_batch_data
                    real_images = real_images.to(self.device)
                    graph_batch_pyg = graph_batch_pyg.to(self.device)
                    current_batch_size = real_images.size(0)
                    if current_batch_size == 0: continue

                    d_real_logits = self.D(real_images)
                    z_graph = self.E(graph_batch_pyg)
                    fake_images = self.G(z_graph, current_batch_size)
                    d_fake_logits = self.D(fake_images)

                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    # Similar to above, R1 is often skipped or made optional for eval.
                    # real_images_for_gp = real_images.clone().requires_grad_(True)
                    # d_real_logits_for_gp = self.D(real_images_for_gp)
                    # r1_penalty_val = self._r1_gradient_penalty(real_images_for_gp, d_real_logits_for_gp)
                    lossD_batch = lossD_adv_batch  # + self.config.r1_gamma * 0.5 * r1_penalty_val

                    d_fake_for_g_logits = self.D(fake_images)
                    lossG_batch = self.loss_fn_g(d_fake_for_g_logits)
                    d_real_logits_mean_batch = d_real_logits.mean()
                    d_fake_logits_mean_batch = d_fake_logits.mean()

                total_d_loss += lossD_batch.item()
                total_g_loss += lossG_batch.item()
                total_d_loss_adv += lossD_adv_batch.item()
                total_r1_penalty += r1_penalty_val.item()
                total_d_real_logits += d_real_logits_mean_batch.item()
                total_d_fake_logits += d_fake_logits_mean_batch.item()
                num_batches += 1

        self.G.train()
        self.D.train()
        if self.E:
            self.E.train()

        if num_batches == 0:
            # This can happen if dataloader is empty due to no files or path not set.
            # get_dataloader now returns None in such cases, handled at the start.
            return {}

        avg_metrics = {
            f"{data_split}/Loss_D": total_d_loss / num_batches,
            f"{data_split}/Loss_G": total_g_loss / num_batches,
            f"{data_split}/Loss_D_Adv": total_d_loss_adv / num_batches,
            f"{data_split}/R1_Penalty_Eval": total_r1_penalty / num_batches,  # Explicitly mark as eval R1
            f"{data_split}/D_Real_Logits_Mean": total_d_real_logits / num_batches,
            f"{data_split}/D_Fake_Logits_Mean": total_d_fake_logits / num_batches,
        }
        print(f"Evaluation results for {data_split}: {avg_metrics}")
        return avg_metrics

    def train(self):
        print("Starting training...")
        eval_freq = getattr(self.config, "eval_freq_epoch", 1)  # Default to evaluating every epoch

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()  # Train one epoch

            # --- Evaluation Phase ---
            if epoch % eval_freq == 0:
                # hasattr check is to ensure config attribute exists (e.g. from BaseConfig)
                if hasattr(self.config, 'dataset_path_val') and self.config.dataset_path_val:
                    val_metrics = self._evaluate_on_split("val")
                    if val_metrics:  # Ensure metrics were actually computed
                        log_to_wandb(self.wandb_run, val_metrics, step=self.current_step)

                if hasattr(self.config, 'dataset_path_test') and self.config.dataset_path_test:
                    test_metrics = self._evaluate_on_split("test")
                    if test_metrics:
                        log_to_wandb(self.wandb_run, test_metrics, step=self.current_step)

            # --- Sample Generation ---
            if epoch % self.config.sample_freq_epoch == 0:
                self.generate_samples(epoch, self.current_step)

            # --- FID Calculation (modified to use test set by default) ---
            if self.config.enable_fid_calculation and \
                    (epoch % self.config.fid_freq_epoch == 0 or epoch == self.config.num_epochs - 1):
                # Default to "test" if available, else "train" for FID real images
                fid_eval_split = "test" if hasattr(self.config,
                                                   'dataset_path_test') and self.config.dataset_path_test else "train"

                # Check if the dataloader for the chosen split can be created
                fid_dataloader_check = get_dataloader(self.config, data_split=fid_eval_split, shuffle=False,
                                                      drop_last=False)
                if fid_dataloader_check is not None:
                    print(f"Calculating FID score using real images from '{fid_eval_split}' split.")
                    fid_score = self.calculate_fid_score(data_split_for_reals=fid_eval_split)
                    if not np.isnan(fid_score):
                        log_to_wandb(self.wandb_run,
                                     {f"FID_Score_{fid_eval_split.capitalize()}": fid_score, "Epoch": epoch},
                                     step=self.current_step)
                else:
                    print(
                        f"Skipping FID calculation: Dataloader for '{fid_eval_split}' split could not be created (path likely not configured or no images).")

            # --- Checkpoint Saving ---
            if epoch % self.config.checkpoint_freq_epoch == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch:04d}.pth.tar")
                checkpoint_data = {
                    'epoch': epoch,
                    'step': self.current_step,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                    'optG_state_dict': self.optG.state_dict(),
                    'optD_state_dict': self.optD.state_dict(),
                    'config': self.config
                }
                if self.model_architecture == "gan6_gat_cnn" and self.E is not None and self.optE is not None:
                    checkpoint_data['E_state_dict'] = self.E.state_dict()
                    checkpoint_data['optE_state_dict'] = self.optE.state_dict()

                save_checkpoint(checkpoint_data, is_best=False, filename=checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        print("Training finished.")
        if self.wandb_run:
            self.wandb_run.finish()

    def _save_images_for_fid(self, images_tensor, base_path, num_images_to_save):
        """Saves a batch of image tensors to a directory for FID calculation."""
        os.makedirs(base_path, exist_ok=True)
        images_tensor = denormalize_image(images_tensor)  # [-1,1] to [0,1]
        for i in range(min(images_tensor.size(0), num_images_to_save)):
            image = images_tensor[i]
            # Convert to PIL Image and save
            # torchvision.transforms.ToPILImage()(image.cpu()).save(os.path.join(base_path, f"img_{i}.png"))
            # Using vutils.save_image for single image:
            vutils.save_image(image.cpu(), os.path.join(base_path, f"img_{i}.png"), normalize=False)

    def calculate_fid_score(self, data_split_for_reals: str = "test"):
        """
        Calculates FID score between generated images and real images from a specified data split.
        Args:
            data_split_for_reals (str): The data split to use for real images ("train", "val", or "test").
                                       This split is also used for conditional inputs to G if needed (e.g. graphs for gan6).
        """
        if not self.config.enable_fid_calculation or calculate_fid_given_paths is None:
            print("FID calculation skipped (either disabled or pytorch-fid not installed).")
            return float('nan')

        print(f"Calculating FID score using real images from '{data_split_for_reals}' split...")
        original_g_mode = self.G.training
        original_e_mode = self.E.training if self.E else None

        self.G.eval()
        if self.E: self.E.eval()

        # Use unique names for FID temp dirs based on the split to avoid conflicts if called multiple times
        # The self.fid_temp_real_path and self.fid_temp_fake_path can be class members for general use,
        # but for a specific run of FID, it's better to make them local to this function or pass them around.
        # For simplicity, let's make them local here.
        current_fid_real_path = os.path.join(self.config.output_dir_run,
                                             f"fid_real_{data_split_for_reals}_epoch{self.current_epoch}")
        current_fid_fake_path = os.path.join(self.config.output_dir_run, f"fid_fake_epoch{self.current_epoch}")

        if os.path.exists(current_fid_real_path): shutil.rmtree(current_fid_real_path)
        if os.path.exists(current_fid_fake_path): shutil.rmtree(current_fid_fake_path)
        os.makedirs(current_fid_real_path, exist_ok=True)
        os.makedirs(current_fid_fake_path, exist_ok=True)

        num_fid_images = self.config.fid_num_images
        # fid_batch_size from config is for Inception model, not GAN generation batch size
        # For GAN generation, we iterate through dataloader with its own batch size.

        # Dataloader for providing conditional inputs to G (if needed) and for saving real images.
        fid_input_dataloader = get_dataloader(self.config, data_split=data_split_for_reals, shuffle=False,
                                              drop_last=False)

        if fid_input_dataloader is None:
            print(
                f"Error: Could not get dataloader for '{data_split_for_reals}' split. Cannot proceed with FID calculation.")
            self.G.train(original_g_mode)
            if self.E and original_e_mode is not None: self.E.train(original_e_mode)
            return float('nan')

        # --- Generate and save fake images ---
        print(f"Generating {num_fid_images} fake images for FID...")
        generated_count = 0
        pbar_fake = tqdm(total=num_fid_images, desc="Generating Fake FID Images")
        for raw_batch_data_fid in fid_input_dataloader:  # Iterate through the chosen split's data
            if generated_count >= num_fid_images: break

            current_batch_actual_size = 0
            if self.model_architecture == "gan5_gcn":
                current_batch_actual_size = raw_batch_data_fid["image"].size(0)
            elif self.model_architecture == "gan6_gat_cnn":
                _, graph_batch_pyg_fid_cond = raw_batch_data_fid
                current_batch_actual_size = graph_batch_pyg_fid_cond.num_graphs if hasattr(graph_batch_pyg_fid_cond,
                                                                                           'num_graphs') else \
                    (graph_batch_pyg_fid_cond.ptr.numel() - 1 if hasattr(graph_batch_pyg_fid_cond, 'ptr') else 0)
            if current_batch_actual_size == 0: continue

            num_to_generate_this_iter = min(current_batch_actual_size, num_fid_images - generated_count)
            if num_to_generate_this_iter <= 0: break

            fake_images_batch = None
            with torch.no_grad():
                if self.model_architecture == "gan5_gcn":
                    real_images_cond = raw_batch_data_fid["image"][:num_to_generate_this_iter].to(self.device)
                    segments_map_cond = raw_batch_data_fid["segments"][:num_to_generate_this_iter].to(self.device)
                    adj_matrix_cond = raw_batch_data_fid["adj"][:num_to_generate_this_iter].to(self.device)
                    z = torch.randn(num_to_generate_this_iter, self.config.model.z_dim, device=self.device)
                    fake_images_batch = self.G(z, real_images_cond, segments_map_cond, adj_matrix_cond)
                elif self.model_architecture == "gan6_gat_cnn":
                    _real_images_input_ignored, graph_batch_pyg_cond = raw_batch_data_fid

                    # Ensure graph_batch_pyg_cond is a PyGBatch object before slicing and sending to device
                    # It might come as a list of Data objects from the dataloader
                    if isinstance(graph_batch_pyg_cond, list) and all(
                            isinstance(g, PyGData) for g in graph_batch_pyg_cond):
                        # Collate into a Batch object first
                        graph_batch_pyg_cond = PyGBatch.from_data_list(graph_batch_pyg_cond)

                    graph_batch_pyg_cond = graph_batch_pyg_cond.to(self.device)

                    # Slicing a PyGBatch object returns another PyGBatch object
                    sliced_graph_batch_cond = graph_batch_pyg_cond[:num_to_generate_this_iter]

                    # Ensure sliced_graph_batch_cond is not empty and is valid before passing to E
                    if not isinstance(sliced_graph_batch_cond, PyGBatch) or sliced_graph_batch_cond.num_graphs == 0:
                        print(
                            f"Warning: Sliced graph batch for FID is invalid or empty (num_graphs: {sliced_graph_batch_cond.num_graphs if isinstance(sliced_graph_batch_cond, PyGBatch) else 'Not a PyGBatch'}). Skipping this iteration for FID.")
                        # fake_images_batch will remain None, handled below
                    else:
                        z_graph = self.E(sliced_graph_batch_cond)
                        fake_images_batch = self.G(z_graph, num_to_generate_this_iter)

            if fake_images_batch is not None:
                for i in range(fake_images_batch.size(0)):
                    if generated_count >= num_fid_images: break
                    img_to_save = denormalize_image(fake_images_batch[i].cpu())
                    vutils.save_image(img_to_save, os.path.join(current_fid_fake_path, f"fake_{generated_count}.png"),
                                      normalize=False)
                    generated_count += 1
                    pbar_fake.update(1)
        pbar_fake.close()
        if generated_count < num_fid_images:
            print(
                f"Warning: Only generated {generated_count}/{num_fid_images} fake images for FID. The '{data_split_for_reals}' dataloader might have provided insufficient conditional data.")

        # --- Save real images from the specified split ---
        print(f"Saving {num_fid_images} real images from '{data_split_for_reals}' split for FID...")
        saved_real_count = 0
        # Re-initialize or reset dataloader for saving real images to ensure it starts from the beginning
        real_dataloader_for_saving = get_dataloader(self.config, data_split=data_split_for_reals, shuffle=False,
                                                    drop_last=False)
        if real_dataloader_for_saving is None:  # Should be caught earlier
            print(f"Critical Error: Dataloader for '{data_split_for_reals}' became None. Skipping FID.")
            self.G.train(original_g_mode);
            if self.E and original_e_mode is not None: self.E.train(original_e_mode)
            return float('nan')

        pbar_real = tqdm(total=num_fid_images, desc=f"Saving Real FID Images ({data_split_for_reals})")
        for raw_batch_data_real_save in real_dataloader_for_saving:
            if saved_real_count >= num_fid_images: break

            real_images_batch_save = raw_batch_data_real_save["image"] if self.model_architecture == "gan5_gcn" else \
                raw_batch_data_real_save[0]
            num_to_save_this_iter = min(real_images_batch_save.size(0), num_fid_images - saved_real_count)

            for i in range(num_to_save_this_iter):
                if saved_real_count >= num_fid_images: break
                img_tensor = denormalize_image(real_images_batch_save[i].cpu())
                vutils.save_image(img_tensor, os.path.join(current_fid_real_path, f"real_{saved_real_count}.png"),
                                  normalize=False)
                saved_real_count += 1
                pbar_real.update(1)
        pbar_real.close()
        if saved_real_count < num_fid_images:
            print(
                f"Warning: Only saved {saved_real_count}/{num_fid_images} real images for FID from '{data_split_for_reals}' split due to dataset size.")

        fid_value = float('nan')
        if generated_count > 0 and saved_real_count > 0:
            # Check for minimum number of images, pytorch-fid might require this
            min_images_for_fid = getattr(self.config, "fid_min_images", 10)  # Add to config if needed, default to 10
            if generated_count >= min_images_for_fid and saved_real_count >= min_images_for_fid:
                try:
                    fid_value = calculate_fid_given_paths(
                        paths=[current_fid_real_path, current_fid_fake_path],
                        batch_size=self.config.fid_batch_size,  # For Inception model
                        device=self.device,
                        dims=2048,
                        num_workers=self.config.num_workers
                    )
                    print(f"FID Score ({data_split_for_reals}): {fid_value:.4f}")
                except Exception as e:
                    print(f"Error calculating FID: {e}")
            else:
                print(
                    f"Skipping FID: Not enough images. Need at least {min_images_for_fid} of each. Got Real: {saved_real_count}, Fake: {generated_count}")
        else:
            print("Not enough images generated or saved for FID calculation. Skipping.")

        # Clean up temporary directories for this specific FID run if desired (or keep for debugging)
        # shutil.rmtree(current_fid_real_path)
        # shutil.rmtree(current_fid_fake_path)

        self.G.train(original_g_mode)
        if self.E and original_e_mode is not None: self.E.train(original_e_mode)
        return fid_value

    def train(self):
        print("Starting training...")
        eval_freq = getattr(self.config, "eval_freq_epoch", 1)  # Default to evaluating every epoch

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.train_epoch()  # Train one epoch

            # --- Evaluation Phase ---
            if epoch % eval_freq == 0:
                if hasattr(self.config, 'dataset_path_val') and self.config.dataset_path_val:
                    val_metrics = self._evaluate_on_split("val")
                    if val_metrics:
                        log_to_wandb(self.wandb_run, val_metrics, step=self.current_step)

                if hasattr(self.config, 'dataset_path_test') and self.config.dataset_path_test:
                    test_metrics = self._evaluate_on_split("test")
                    if test_metrics:
                        log_to_wandb(self.wandb_run, test_metrics, step=self.current_step)

            # --- Sample Generation ---
            if epoch % self.config.sample_freq_epoch == 0:
                self.generate_samples(epoch, self.current_step)

            # --- FID Calculation ---
            if self.config.enable_fid_calculation and \
                    (epoch % self.config.fid_freq_epoch == 0 or epoch == self.config.num_epochs - 1):
                fid_eval_split_for_reals = "test" if hasattr(self.config,
                                                             'dataset_path_test') and self.config.dataset_path_test else "train"

                fid_score = self.calculate_fid_score(data_split_for_reals=fid_eval_split_for_reals)
                if not np.isnan(fid_score):
                    log_to_wandb(self.wandb_run,
                                 {f"FID_Score_{fid_eval_split_for_reals.capitalize()}": fid_score, "Epoch": epoch},
                                 step=self.current_step)

            # --- Checkpoint Saving ---
            if epoch % self.config.checkpoint_freq_epoch == 0 or epoch == self.config.num_epochs - 1:
                checkpoint_path = os.path.join(self.checkpoints_dir, f"checkpoint_epoch_{epoch:04d}.pth.tar")
                save_checkpoint({
                    'epoch': epoch,
                    'step': self.current_step,
                    'G_state_dict': self.G.state_dict(),
                    'D_state_dict': self.D.state_dict(),
                    'optG_state_dict': self.optG.state_dict(),
                    'optD_state_dict': self.optD.state_dict(),
                    'config': self.config  # Save config with checkpoint
                }, is_best=False,
                    filename=checkpoint_path)  # 'is_best' logic can be added if there's a validation metric
                print(f"Checkpoint saved to {checkpoint_path}")

        print("Training finished.")
        if self.wandb_run:
            self.wandb_run.finish()


print("src/trainer.py created with Trainer class.")
