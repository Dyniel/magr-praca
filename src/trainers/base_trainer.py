import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
import abc

from src.data_loader import get_dataloader
from src.utils import (
    denormalize_image,
    generate_spatial_superpixel_map,
    calculate_mean_superpixel_features,
    toggle_grad,
    log_image_grid_to_wandb,
    interpolate_spatial_map,
)
from src.models.superpixel_encoder import SuperpixelLatentEncoder
from src.losses.adversarial import r1_penalty
from src.metrics import calculate_fid

class BaseTrainer(abc.ABC):
    def __init__(self, config):
        self.config = config
        self._init_wandb()

        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA specified in config but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)
        print(f"Using device: {self.device}")

        self.model_architecture = self.config.model.architecture
        self.current_epoch = 0
        self.current_iteration = 0
        self.ada_manager = None

        self._init_models()
        self._init_optimizers()
        self._init_loss_functions()
        self._watch_models()

    def _init_wandb(self):
        if hasattr(self.config, 'logging') and self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project_name,
                entity=self.config.logging.wandb_entity,
                name=self.config.logging.wandb_run_name,
                config=OmegaConf.to_container(self.config, resolve=True)
            )
            # Initialize a wandb.Table for logging sample predictions
            self.sample_predictions_table = wandb.Table(columns=["epoch", "step", "image"])

    def _watch_models(self):
        if wandb.run is not None:
            if hasattr(self.config, 'logging'):
                if hasattr(self, 'G') and self.G is not None:
                    wandb.watch(self.G, log="all", log_freq=self.config.logging.wandb_watch_freq_g)
                if hasattr(self, 'D') and self.D is not None:
                    wandb.watch(self.D, log="all", log_freq=self.config.logging.wandb_watch_freq_d)

    @abc.abstractmethod
    def _init_models(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_optimizers(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_loss_functions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _train_d(self, real_images, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def _train_g(self, real_images, **kwargs):
        raise NotImplementedError

    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs...")
        train_dataloader = get_dataloader(self.config, data_split="train", shuffle=True, drop_last=True)
        if train_dataloader is None:
            print("No training dataloaloader found. Exiting.")
            return

        log_freq_step = max(1, len(train_dataloader) // 10)
        print(f"Logging metrics every {log_freq_step} steps.")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            self.G.train()
            self.D.train()
            if hasattr(self, 'E') and self.E: self.E.train()
            if hasattr(self, 'sp_latent_encoder') and self.sp_latent_encoder: self.sp_latent_encoder.train()

            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, raw_batch_data in enumerate(batch_iterator):
                if raw_batch_data is None:
                    print(f"Warning: Trainer received a None batch from dataloader at training iteration {self.current_iteration}. Skipping batch.")
                    self.current_iteration +=1
                    continue
                self.current_iteration += 1
                logs = {}

                real_images_gan_norm, segments_map, adj_matrix, graph_batch_pyg = self._unpack_batch(raw_batch_data)
                if real_images_gan_norm is None or real_images_gan_norm.size(0) == 0:
                    print(f"Warning: Could not extract real images for training batch or batch is empty. Skipping.")
                    continue

                # --- Discriminator Training ---
                if self.current_iteration % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_D.zero_grad()

                d_logs = self._train_d(real_images_gan_norm, segments_map=segments_map, adj_matrix=adj_matrix, graph_batch_pyg=graph_batch_pyg)
                logs.update(d_logs)

                # --- Generator Training ---
                if self.current_iteration % self.config.d_updates_per_g_update == 0:
                    if self.current_iteration % self.config.gradient_accumulation_steps == 0:
                        self.optimizer_G.zero_grad()

                    g_logs = self._train_g(real_images_gan_norm, segments_map=segments_map, adj_matrix=adj_matrix, graph_batch_pyg=graph_batch_pyg)
                    logs.update(g_logs)

                if self.current_iteration % log_freq_step == 0:
                    batch_iterator.set_postfix(logs)
                    if self.config.logging.use_wandb and wandb.run:
                        wandb.log(logs, step=self.current_iteration)

                # ... (sample logging and checkpointing logic will be added here)

            self.G.eval()
            with torch.no_grad():
                # Generate a grid of images for a general overview
                z_noise_grid = torch.randn(self.config.batch_size, self.config.model.stylegan2_z_dim, device=self.device)
                g_kwargs = {
                    'style_mix_prob': getattr(self.config.model, 'stylegan2_style_mix_prob', 0.9),
                    'truncation_psi': None
                }
                fake_images_grid = self.G(z_noise_grid, **g_kwargs)
                if self.config.logging.use_wandb and wandb.run:
                    log_image_grid_to_wandb(denormalize_image(fake_images_grid), wandb.run, f"Generated Images Epoch {epoch+1}", self.current_iteration)

                # Log 5 sample predictions to the table
                print("Logging 5 sample predictions to wandb table...")
                z_noise_samples = torch.randn(5, self.config.model.stylegan2_z_dim, device=self.device)
                sample_images = self.G(z_noise_samples, **g_kwargs)
                denormalized_samples = denormalize_image(sample_images)
                if self.config.logging.use_wandb and wandb.run:
                    for i in range(denormalized_samples.size(0)):
                        self.sample_predictions_table.add_data(
                            self.current_epoch,
                            self.current_iteration,
                            wandb.Image(denormalized_samples[i])
                        )

                # Calculate FID
                if self.config.logging.calculate_fid:
                    print("Calculating FID...")
                    # Generate a larger batch of fake images for FID calculation
                    z_noise_fid = torch.randn(self.config.logging.fid_num_images, self.config.model.stylegan2_z_dim, device=self.device)
                    fake_images_fid = self.G(z_noise_fid, **g_kwargs)

                    # Get a batch of real images
                    real_images_fid, _, _, _ = self._unpack_batch(next(iter(train_dataloader)))

                    # Denormalize images for FID calculation
                    fake_images_fid_denorm = denormalize_image(fake_images_fid)
                    real_images_fid_denorm = denormalize_image(real_images_fid)

                    fid_score = calculate_fid(real_images_fid_denorm, fake_images_fid_denorm, self.config.batch_size, self.device.type == 'cuda')
                    if self.config.logging.use_wandb and wandb.run:
                        wandb.log({"FID": fid_score}, step=self.current_iteration)
                    print(f"FID at epoch {epoch+1}: {fid_score}")


            self.G.train()

            print(f"Epoch {epoch+1} completed.")

        print("Training finished.")
        if hasattr(self.config, 'logging') and self.config.logging.use_wandb and wandb.run:
            wandb.log({"sample_predictions": self.sample_predictions_table})
            wandb.finish()
        elif getattr(self.config, 'use_wandb', False) and wandb.run:
            wandb.log({"sample_predictions": self.sample_predictions_table})
            wandb.finish()

    def _unpack_batch(self, raw_batch_data):
        real_images_gan_norm, segments_map, adj_matrix, graph_batch_pyg = None, None, None, None
        if isinstance(raw_batch_data, dict) and "image" in raw_batch_data:
            real_images_gan_norm = raw_batch_data["image"].to(self.device)
            if "segments" in raw_batch_data: segments_map = raw_batch_data["segments"].to(self.device)
            if "adj" in raw_batch_data: adj_matrix = raw_batch_data["adj"].to(self.device)
        elif isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2:
            real_images_gan_norm, graph_batch_pyg = raw_batch_data
            real_images_gan_norm = real_images_gan_norm.to(self.device)
            graph_batch_pyg = graph_batch_pyg.to(self.device)
        elif isinstance(raw_batch_data, torch.Tensor):
            real_images_gan_norm = raw_batch_data.to(self.device)
        return real_images_gan_norm, segments_map, adj_matrix, graph_batch_pyg

    def _prepare_superpixel_conditioning(self, segments_map, real_images_gan_norm):
        spatial_map_g, spatial_map_d, z_superpixel_g = None, None, None
        g_spatial_active = getattr(self.config.model, f"{self.model_architecture}_g_spatial_cond", False)
        d_spatial_active = getattr(self.config.model, f"{self.model_architecture}_d_spatial_cond", False)
        g_latent_active = hasattr(self, 'sp_latent_encoder') and self.sp_latent_encoder is not None and \
                          getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False)

        if self.config.model.use_superpixel_conditioning and segments_map is not None and \
           (g_spatial_active or d_spatial_active or g_latent_active):
            real_images_01 = denormalize_image(real_images_gan_norm)

            if g_spatial_active:
                spatial_map_g = generate_spatial_superpixel_map(
                    segments_map, self.config.model.superpixel_spatial_map_channels_g,
                    self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                spatial_map_g = interpolate_spatial_map(spatial_map_g, size=4)

            if d_spatial_active:
                spatial_map_d = generate_spatial_superpixel_map(
                segments_map, self.config.model.superpixel_spatial_map_channels_d,
                self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)

            if g_latent_active:
                mean_sp_feats = calculate_mean_superpixel_features(
                    real_images_01, segments_map,
                    self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(
                    self.device)
                z_superpixel_g = self.sp_latent_encoder(mean_sp_feats)
        return spatial_map_g, spatial_map_d, z_superpixel_g