import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import wandb # Assuming wandb is used for logging, can be made optional

# Project specific imports - these might need adjustment based on actual file structure and names
from src.models import (
    Generator as GAN5Generator, Discriminator as GAN5Discriminator, # gan5_gcn
    GraphEncoderGAT, GeneratorCNN as GAN6Generator, DiscriminatorCNN as GAN6Discriminator, # gan6_gat_cnn
    DCGANGenerator, DCGANDiscriminator,
    StyleGAN2Generator, StyleGAN2Discriminator,
    StyleGAN3Generator, StyleGAN3Discriminator, # Assuming these exist
    ProjectedGANGenerator, ProjectedGANDiscriminator, FeatureExtractor, # Assuming these exist
    SuperpixelLatentEncoder
)
from src.data_loader import get_dataloader
from src.utils import (
    denormalize_image, generate_spatial_superpixel_map, calculate_mean_superpixel_features,
    toggle_grad, compute_grad_penalty # R1 gradient penalty
)
# Add other necessary loss functions or utilities
# from src.losses import generator_loss_nonsaturating, discriminator_loss_r1, ...


class Trainer:
    def __init__(self, config): # Parameter name changed back to config
        # Log the OmegaConf object passed as 'config'
        if hasattr(config, 'logging') and config.logging.use_wandb and config.logging.wandb_project_name:
            wandb.init(
                project=config.logging.wandb_project_name,
                entity=config.logging.wandb_entity,
                name=config.logging.wandb_run_name,
                config=OmegaConf.to_container(config, resolve=True) # Log the original OmegaConf
            )
        elif hasattr(config, 'use_wandb') and config.use_wandb: # Fallback
            print("Warning: 'logging' attribute not found in config, but 'use_wandb' is true. Attempting legacy WandB init.")
            wandb.init(
                project=getattr(config, 'wandb_project_name', 'default_project'),
                name=getattr(config, 'wandb_run_name', 'default_run'),
                config=OmegaConf.to_container(config, resolve=True)
            )

        # Convert the incoming OmegaConf 'config' object to the actual BaseConfig dataclass instance for internal use
        # This instance will be stored in self.config, shadowing the parameter name, which is fine.
        self.config = OmegaConf.to_object(config)

        # Updated device selection logic using the dataclass instance self.config
        if self.config.device == "cuda" and not torch.cuda.is_available():
            print("CUDA specified in config but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"Using device: {self.device}")

        self.model_architecture = self.config.model.architecture
        self.current_epoch = 0
        self.current_iteration = 0

        # Initialize models, optimizers, loss functions based on self.config (which is now BaseConfig instance)
        self._init_models()
        self._init_optimizers()
        self._init_loss_functions()

        # For ProjectedGAN feature matching
        if self.model_architecture == "projected_gan":
            self.feature_extractor = FeatureExtractor(
                model_name=self.config.model.projectedgan_feature_extractor_model,
                layers_to_extract=self.config.model.projectedgan_feature_extractor_layers,
                pretrained=True,
                requires_grad=False
            ).to(self.device).eval()
            # self.imagenet_norm = ...

        # WandB watch calls (moved after G and D are initialized in _init_models)
        # Only call wandb.watch if wandb.init was successful (i.e., wandb.run is not None)
        if wandb.run is not None:
            if hasattr(self.config, 'logging'): # Check if logging config exists
                if hasattr(self, 'G') and self.G is not None:
                    wandb.watch(self.G, log="all", log_freq=self.config.logging.wandb_watch_freq_g)
                if hasattr(self, 'D') and self.D is not None:
                    wandb.watch(self.D, log="all", log_freq=self.config.logging.wandb_watch_freq_d)
            elif hasattr(self.config, 'use_wandb') and self.config.use_wandb: # Fallback for older config structure
                 if hasattr(self, 'G') and self.G is not None: wandb.watch(self.G, log="all")
                 if hasattr(self, 'D') and self.D is not None: wandb.watch(self.D, log="all")


    def _init_models(self):
        # Generator and Discriminator
        if self.model_architecture == "gan5_gcn":
            self.G = GAN5Generator(self.config).to(self.device)
            self.D = GAN5Discriminator(self.config).to(self.device)
            self.E = None # gan5_gcn doesn't use a separate E model in this structure
        elif self.model_architecture == "gan6_gat_cnn":
            self.G = GAN6Generator(self.config).to(self.device)
            self.D = GAN6Discriminator(self.config).to(self.device)
            self.E = GraphEncoderGAT(self.config).to(self.device) if self.config.model.gan6_use_graph_encoder else None
        elif self.model_architecture == "dcgan":
            self.G = DCGANGenerator(self.config).to(self.device)
            self.D = DCGANDiscriminator(self.config).to(self.device)
            self.E = None
        elif self.model_architecture == "stylegan2":
            self.G = StyleGAN2Generator(self.config).to(self.device)
            self.D = StyleGAN2Discriminator(self.config).to(self.device)
            self.E = None
            # StyleGAN2 specific: w_avg for truncation trick
            self.w_avg = None
            if self.config.model.stylegan2_use_truncation:
                 # Typically initialized by running G with many z and averaging w.
                 # Placeholder: self.w_avg = torch.zeros(self.config.model.stylegan2_w_dim, device=self.device)
                 pass # This would be calculated later or loaded.
        elif self.model_architecture == "stylegan3":
            self.G = StyleGAN3Generator(self.config).to(self.device)
            self.D = StyleGAN3Discriminator(self.config).to(self.device)
            self.E = None
        elif self.model_architecture == "projected_gan":
            self.G = ProjectedGANGenerator(self.config).to(self.device) # Based on StyleGAN2
            self.D = ProjectedGANDiscriminator(self.config).to(self.device)
            self.E = None
            # self.w_avg for G if it's StyleGAN2 based and uses truncation
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        # Superpixel Latent Encoder (C2 conditioning)
        self.sp_latent_encoder = None
        if self.config.model.use_superpixel_conditioning and \
           getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False):
            self.sp_latent_encoder = SuperpixelLatentEncoder(
                input_feature_dim=self.config.model.superpixel_feature_dim, # e.g., 3 for RGB
                hidden_dims=self.config.model.superpixel_latent_encoder_hidden_dims,
                output_embedding_dim=self.config.model.superpixel_latent_embedding_dim,
                num_superpixels=self.config.num_superpixels
            ).to(self.device)

        print("Models initialized:")
        print(f"G: {self.G.__class__.__name__}, D: {self.D.__class__.__name__}")
        if self.E: print(f"E: {self.E.__class__.__name__}")
        if self.sp_latent_encoder: print(f"SP_Encoder: {self.sp_latent_encoder.__class__.__name__}")


    def _init_optimizers(self):
        g_params = list(self.G.parameters())
        if self.E: g_params += list(self.E.parameters())
        if self.sp_latent_encoder: g_params += list(self.sp_latent_encoder.parameters())

        self.optimizer_G = optim.Adam(
            g_params,
            lr=self.config.optimizer.g_lr, # Updated path
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2) # Updated path
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=self.config.optimizer.d_lr, # Updated path
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2) # Updated path

        )
        print("Optimizers initialized.")

    def _init_loss_functions(self):
        # Define loss functions based on model architecture or config
        # This is a simplified example. Specific GANs have specific loss formulations.

        # r1_gamma is now expected to be a top-level parameter in BaseConfig,
        # as sweeps are setting it directly.
        self.r1_gamma = self.config.r1_gamma


        if self.model_architecture in ["gan5_gcn", "gan6_gat_cnn", "dcgan"]:
            # Standard GAN losses (non-saturating or LSGAN, etc.)
            self.loss_fn_g = lambda d_fake_logits: F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
            self.loss_fn_d = lambda d_real_logits, d_fake_logits: \
                F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits)) + \
                F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            # self.r1_gamma is already set from top-level config
        elif self.model_architecture == "stylegan2":
            self.loss_fn_g_stylegan2 = lambda d_fake_logits: F.softplus(-d_fake_logits).mean() # Non-saturating
            self.loss_fn_d_stylegan2 = lambda d_real_logits, d_fake_logits: \
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
            # self.r1_gamma is set from top-level config. If model-specific r1 is needed, logic would change.
            # Example: self.r1_gamma = self.config.model.stylegan2_r1_gamma if hasattr(self.config.model, 'stylegan2_r1_gamma') else self.config.r1_gamma

        elif self.model_architecture == "stylegan3":
            self.loss_fn_g_stylegan3 = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_stylegan3 = lambda d_real_logits, d_fake_logits: \
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
            # self.r1_gamma from top-level
        elif self.model_architecture == "projected_gan":
            self.loss_fn_g_adv_projected = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_adv_projected = lambda d_real_logits, d_fake_logits: \
                 F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean() # Or hinge
            self.loss_fn_g_feat_match = nn.MSELoss()
            # self.r1_gamma from top-level
        else:
            self.loss_fn_g = None
            self.loss_fn_d = None
            # self.r1_gamma will use the value from top-level config, or its default if not overridden by sweep.

        print(f"Loss functions initialized. R1 Gamma set to: {self.r1_gamma}")


    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs...") # Use self.config.num_epochs

        train_dataloader = get_dataloader(self.config, data_split="train", shuffle=True, drop_last=True)
        if train_dataloader is None:
            print("No training dataloader found. Exiting.")
            return

        for epoch in range(self.current_epoch, self.config.num_epochs): # Use self.config.num_epochs

            self.current_epoch = epoch
            self.G.train()
            self.D.train()
            if self.E: self.E.train()
            if self.sp_latent_encoder: self.sp_latent_encoder.train()

            # Corrected: self.config.num_epochs instead of self.config.training.epochs
            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

            epoch_losses_d_adv = []
            epoch_losses_d_r1 = []
            epoch_losses_d_total = []
            epoch_losses_g_adv = []
            epoch_losses_g_feat = []
            epoch_losses_g_total = []
            processed_batches_in_epoch = 0

            for batch_idx, raw_batch_data in enumerate(batch_iterator):
                if raw_batch_data is None:
                    print(f"Warning: Trainer received a None batch from dataloader at training iteration {self.current_iteration} (epoch {epoch+1}, batch_idx {batch_idx}). Skipping batch.")
                    # self.current_iteration is incremented per batch attempt later, so don't double count here unless strictly iteration based
                    continue

                self.current_iteration +=1 # Increment for each attempted batch
                current_batch_logs = {} # Renamed from logs to avoid conflict with epoch logs


                # --- Prepare inputs for current batch (real images, segments, etc.) ---
                # This logic is similar to _evaluate_on_split, adapted for training
                real_images_gan_norm = None; segments_map = None; adj_matrix = None; graph_batch_pyg = None
                if isinstance(raw_batch_data, dict) and "image" in raw_batch_data:
                    real_images_gan_norm = raw_batch_data["image"].to(self.device)
                    if "segments" in raw_batch_data: segments_map = raw_batch_data["segments"].to(self.device)
                    if "adj" in raw_batch_data: adj_matrix = raw_batch_data["adj"].to(self.device)
                elif isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2: # For graph-based GANs like gan6
                    real_images_gan_norm, graph_batch_pyg = raw_batch_data
                    real_images_gan_norm = real_images_gan_norm.to(self.device)
                    graph_batch_pyg = graph_batch_pyg.to(self.device)
                elif isinstance(raw_batch_data, torch.Tensor): # Simple image tensor
                    real_images_gan_norm = raw_batch_data.to(self.device)

                if real_images_gan_norm is None:
                    print(f"Warning: Could not extract real images for training batch. Skipping.")
                    continue
                current_batch_size = real_images_gan_norm.size(0)
                if current_batch_size == 0: continue

                # --- Prepare Superpixel Conditioning Tensors (if enabled) ---
                spatial_map_g, spatial_map_d, z_superpixel_g = None, None, None
                g_spatial_active = getattr(self.config.model, f"{self.model_architecture}_g_spatial_cond", False)
                d_spatial_active = getattr(self.config.model, f"{self.model_architecture}_d_spatial_cond", False)
                g_latent_active = self.sp_latent_encoder is not None and \
                                  getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False)

                if self.config.model.use_superpixel_conditioning and segments_map is not None and \
                   (g_spatial_active or d_spatial_active or g_latent_active):
                    real_images_01 = denormalize_image(real_images_gan_norm) # For feature/map generation
                    if g_spatial_active:
                        spatial_map_g = generate_spatial_superpixel_map(
                            segments_map, self.config.model.superpixel_spatial_map_channels_g,
                            self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                        # Resize if G expects a fixed small map (e.g. StyleGANs starting at 4x4)
                        if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan", "dcgan"] and \
                           hasattr(self.config.model, "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                           spatial_map_g is not None and spatial_map_g.shape[-1] != 4: # Assuming G starts at 4x4
                             spatial_map_g = F.interpolate(spatial_map_g, size=(4,4), mode='nearest')

                    if d_spatial_active: # For D, use map from real image's segments
                        spatial_map_d = generate_spatial_superpixel_map(
                            segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                            # D usually takes full-res spatial map, no resize needed here unless specific D arch.
                    if g_latent_active:
                        mean_sp_feats = calculate_mean_superpixel_features(
                            real_images_01, segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(self.device)
                        z_superpixel_g = self.sp_latent_encoder(mean_sp_feats)


                # --- Train Discriminator ---
                toggle_grad(self.D, True)
                self.optimizer_D.zero_grad()

                # Real images
                real_images_gan_norm.requires_grad = (self.r1_gamma > 0) # For R1 penalty
                d_real_logits = self.D(real_images_gan_norm, spatial_map_d=spatial_map_d)

                # Fake images
                z_noise = torch.randn(current_batch_size, self.config.model.get(f"{self.model_architecture}_z_dim", self.config.model.z_dim), device=self.device) # Get correct z_dim

                # Generator forward pass arguments vary by architecture
                g_args = [z_noise]
                g_kwargs = {}
                if self.model_architecture == "gan5_gcn":
                    g_args.extend([real_images_gan_norm, segments_map, adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    z_graph = self.E(graph_batch_pyg)
                    if self.config.model.gan6_gat_cnn_use_null_graph_embedding: z_graph = torch.zeros_like(z_graph)
                    g_args = [z_graph, current_batch_size] # GAN6 G takes z_graph and batch_size
                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    # These models can take additional conditioning arguments
                    g_kwargs['spatial_map_g'] = spatial_map_g
                    g_kwargs['z_superpixel_g'] = z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]:
                        g_kwargs['style_mix_prob'] = self.config.model.get('stylegan2_style_mix_prob', 0.9)
                        g_kwargs['truncation_psi'] = None # No truncation during training typically
                    # Add other specific args if needed, like input_is_w for StyleGANs if doing w-space training

                with torch.no_grad(): # Detach G's output for D training
                    fake_images = self.G(*g_args, **g_kwargs)
                d_fake_logits = self.D(fake_images.detach(), spatial_map_d=spatial_map_d) # Pass spatial_map_d also for fake if D is conditioned

                # Discriminator loss
                lossD_adv_val = 0
                lossD_r1_val = 0
                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits)
                else:
                    lossD = self.loss_fn_d(d_real_logits, d_fake_logits)
                lossD_adv_val = lossD.item()
                current_batch_logs["Loss_D_Adv"] = lossD_adv_val

                if self.r1_gamma > 0:
                    r1_penalty = compute_grad_penalty(d_real_logits, real_images_gan_norm) * self.r1_gamma / 2
                    lossD += r1_penalty
                    lossD_r1_val = r1_penalty.item()
                    current_batch_logs["Loss_D_R1"] = lossD_r1_val

                lossD_total_val = lossD.item()
                current_batch_logs["Loss_D_Total"] = lossD_total_val

                epoch_losses_d_adv.append(lossD_adv_val)
                if self.r1_gamma > 0: epoch_losses_d_r1.append(lossD_r1_val) # Only append if calculated
                epoch_losses_d_total.append(lossD_total_val)

                lossD.backward()
                self.optimizer_D.step()
                toggle_grad(self.D, False)

                # --- Train Generator ---
                lossG_adv_val = 0
                lossG_feat_val = 0
                lossG_total_val = 0

                if self.current_iteration % self.config.d_updates_per_g_update == 0:
                    toggle_grad(self.G, True)
                    if self.E: toggle_grad(self.E, True)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, True)
                    self.optimizer_G.zero_grad()

                    z_noise_g = torch.randn(current_batch_size, self.config.model.get(f"{self.model_architecture}_z_dim", self.config.model.z_dim), device=self.device)
                    g_args_g = [z_noise_g]
                    g_kwargs_g = {}
                    if self.model_architecture == "gan5_gcn":
                        g_args_g.extend([real_images_gan_norm, segments_map, adj_matrix])
                    elif self.model_architecture == "gan6_gat_cnn":
                        z_graph_g = self.E(graph_batch_pyg) if self.E else torch.zeros((current_batch_size, self.config.model.gan6_z_dim_graph_encoder_output), device=self.device)
                        if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E: z_graph_g = torch.zeros_like(z_graph_g)
                        g_args_g = [z_graph_g, current_batch_size]
                    elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                        g_kwargs_g['spatial_map_g'] = spatial_map_g
                        g_kwargs_g['z_superpixel_g'] = z_superpixel_g
                        if self.model_architecture in ["stylegan2", "projected_gan"]:
                            g_kwargs_g['style_mix_prob'] = self.config.model.get('stylegan2_style_mix_prob', 0.9)

                    fake_images_for_g = self.G(*g_args_g, **g_kwargs_g)
                    d_fake_logits_for_g = self.D(fake_images_for_g, spatial_map_d=spatial_map_d)

                    if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                        lossG_adv = self.loss_fn_g_stylegan2(d_fake_logits_for_g)
                    else:
                        lossG_adv = self.loss_fn_g(d_fake_logits_for_g)
                    lossG_adv_val = lossG_adv.item()
                    current_batch_logs["Loss_G_Adv"] = lossG_adv_val
                    lossG = lossG_adv

                    if self.model_architecture == "projected_gan":
                        real_01 = denormalize_image(real_images_gan_norm)
                        fake_01 = denormalize_image(fake_images_for_g)
                        real_feats_dict = self.feature_extractor(real_01)
                        fake_feats_dict = self.feature_extractor(fake_01)
                        lossG_feat_single_batch = 0.0 # Initialize as float
                        temp_loss_val = 0.0
                        for key_idx, key in enumerate(real_feats_dict):
                             # Ensure features are tensors before loss calculation
                            if isinstance(fake_feats_dict[key], torch.Tensor) and isinstance(real_feats_dict[key], torch.Tensor):
                                temp_loss_val += self.loss_fn_g_feat_match(fake_feats_dict[key], real_feats_dict[key].detach())
                        if isinstance(temp_loss_val, torch.Tensor): # if it became a tensor
                            lossG_feat_single_batch = temp_loss_val.item()
                        else: # if it remained float (e.g. no features)
                            lossG_feat_single_batch = temp_loss_val

                        lossG += self.config.model.projectedgan_feature_matching_loss_weight * temp_loss_val # temp_loss_val is tensor or float
                        lossG_feat_val = lossG_feat_single_batch # Already .item() or float
                        current_batch_logs["Loss_G_FeatMatch"] = lossG_feat_val

                    lossG_total_val = lossG.item() # This should be after all additions to lossG
                    current_batch_logs["Loss_G_Total"] = lossG_total_val

                    epoch_losses_g_adv.append(lossG_adv_val)
                    if self.model_architecture == "projected_gan": epoch_losses_g_feat.append(lossG_feat_val)
                    epoch_losses_g_total.append(lossG_total_val)

                    lossG.backward()
                    self.optimizer_G.step()

                    toggle_grad(self.G, False)
                    if self.E: toggle_grad(self.E, False)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, False)

                processed_batches_in_epoch += 1

                # Per-step logging
                if hasattr(self.config, 'logging') and self.current_iteration % self.config.logging.log_freq_step == 0:
                    batch_iterator.set_postfix(current_batch_logs)
                    if self.config.logging.use_wandb and wandb.run:
                        wandb.log(current_batch_logs, step=self.current_iteration)
                elif self.current_iteration % getattr(self.config, 'log_freq_step', 100) == 0:
                    batch_iterator.set_postfix(current_batch_logs)
                    if getattr(self.config, 'use_wandb', False) and wandb.run:
                         wandb.log(current_batch_logs, step=self.current_iteration)

            # --- End of Epoch ---
            if processed_batches_in_epoch > 0:
                epoch_summary_logs = {"epoch": epoch + 1}
                # Calculate means, ensuring lists are not empty to avoid division by zero if using np.mean directly
                if epoch_losses_d_total: epoch_summary_logs["train_epoch/Loss_D_Total"] = sum(epoch_losses_d_total) / len(epoch_losses_d_total)
                if epoch_losses_d_adv: epoch_summary_logs["train_epoch/Loss_D_Adv"] = sum(epoch_losses_d_adv) / len(epoch_losses_d_adv)
                if epoch_losses_d_r1: epoch_summary_logs["train_epoch/Loss_D_R1"] = sum(epoch_losses_d_r1) / len(epoch_losses_d_r1)

                if epoch_losses_g_total: epoch_summary_logs["train_epoch/Loss_G_Total"] = sum(epoch_losses_g_total) / len(epoch_losses_g_total)
                if epoch_losses_g_adv: epoch_summary_logs["train_epoch/Loss_G_Adv"] = sum(epoch_losses_g_adv) / len(epoch_losses_g_adv)
                if epoch_losses_g_feat: epoch_summary_logs["train_epoch/Loss_G_FeatMatch"] = sum(epoch_losses_g_feat) / len(epoch_losses_g_feat)

                print(f"\n--- Epoch {epoch+1}/{self.config.num_epochs} Summary (Train) ---")
                for key, value in epoch_summary_logs.items():
                    if key != "epoch": print(f"{key}: {value:.4f}")

                if hasattr(self.config, 'logging') and self.config.logging.use_wandb and wandb.run:
                    wandb.log(epoch_summary_logs, step=self.current_iteration)
            else:
                print(f"Epoch {epoch+1}: No batches were processed. Skipping epoch summary logging for training.")

            # Evaluation (e.g., on validation set), sample logging, and checkpointing
            # These are typically done at the end of an epoch or every N epochs.
            if hasattr(self.config, 'logging'):
                # Log samples and validation metrics if sample_freq_epoch is met (e.g., every N epochs)
                if self.config.logging.sample_freq_epoch > 0 and (epoch + 1) % self.config.logging.sample_freq_epoch == 0:
                    eval_metrics = self._evaluate_on_split("val") # Assuming this handles its own sample logging to wandb
                    if self.config.logging.use_wandb and wandb.run and eval_metrics:
                         # eval_metrics from _evaluate_on_split already have "val/" prefix
                        wandb.log(eval_metrics, step=self.current_iteration)

                # Checkpointing
                if (epoch + 1) % self.config.logging.checkpoint_freq_epoch == 0 :
                    self.save_checkpoint(epoch=self.current_epoch, iteration=self.current_iteration, is_best=False)
            # Fallback checkpointing if logging object not fully configured
            elif (epoch + 1) % getattr(self.config, 'checkpoint_freq_epoch', 10) == 0:
                self.save_checkpoint(epoch=self.current_epoch, iteration=self.current_iteration, is_best=False)

            # FID Calculation
            if self.config.enable_fid_calculation and (epoch + 1) % self.config.fid_freq_epoch == 0:
                print(f"Epoch {epoch+1}: FID calculation would occur here (not yet implemented).")
                # fid_score = self.calculate_fid()
                # if fid_score is not None and hasattr(self.config, 'logging') and self.config.logging.use_wandb and wandb.run:
                #     wandb.log({"val/FID_Score": fid_score}, step=self.current_iteration)

            print(f"--- Epoch {epoch+1} processing completed. ---")

        print("Training finished.")
        # Corrected: Check for logging attribute and use self.config.logging.use_wandb
        if hasattr(self.config, 'logging') and self.config.logging.use_wandb:
            wandb.finish()
        elif getattr(self.config, 'use_wandb', False): # Fallback
            wandb.finish()

    def _evaluate_on_split(self, data_split: str):
        """
        Evaluates the model on a given data split (e.g., "val" or "test").
        Calculates losses and other relevant metrics.
        """
        print(f"Evaluating on {data_split} split...")
        eval_dataloader = get_dataloader(self.config, data_split=data_split, shuffle=False, drop_last=False)

        if eval_dataloader is None:
            print(f"No dataloader for {data_split} split (path likely not configured). Skipping evaluation.")
            return {}

        self.G.eval()
        self.D.eval()
        if self.E: self.E.eval()
        if self.sp_latent_encoder: self.sp_latent_encoder.eval()

        total_d_loss = 0.0
        total_g_loss = 0.0
        total_d_loss_adv = 0.0
        # total_r1_penalty = 0.0 # R1 is typically not computed/enforced during eval
        total_d_real_logits = 0.0
        total_d_fake_logits = 0.0
        total_g_feat_match_loss = 0.0 # For ProjectedGAN
        num_batches = 0

        with torch.no_grad():
            for batch_idx, raw_batch_data in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {data_split}")):
                if raw_batch_data is None:
                    print(f"Warning: Trainer received a None batch from dataloader during {data_split} evaluation (batch_idx {batch_idx}). Skipping batch.")
                    continue

                # Initialize batch losses/metrics to tensor(0.0) on correct device
                lossD_batch = torch.tensor(0.0, device=self.device)
                lossG_batch = torch.tensor(0.0, device=self.device)
                lossD_adv_batch = torch.tensor(0.0, device=self.device)
                d_real_logits_mean_batch = torch.tensor(0.0, device=self.device)
                d_fake_logits_mean_batch = torch.tensor(0.0, device=self.device)
                lossG_feat_match_batch = torch.tensor(0.0, device=self.device)
                current_batch_size = 0

                # --- Prepare inputs for current batch (real images, segments, etc.) ---
                eval_real_images_gan_norm = None; eval_segments_map = None; eval_adj_matrix = None; eval_graph_batch_pyg = None
                if isinstance(raw_batch_data, dict) and "image" in raw_batch_data:
                    eval_real_images_gan_norm = raw_batch_data["image"].to(self.device)
                    if "segments" in raw_batch_data: eval_segments_map = raw_batch_data["segments"].to(self.device)
                    if "adj" in raw_batch_data: eval_adj_matrix = raw_batch_data["adj"].to(self.device)
                elif isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2:
                    eval_real_images_gan_norm, eval_graph_batch_pyg = raw_batch_data
                    eval_real_images_gan_norm = eval_real_images_gan_norm.to(self.device)
                    eval_graph_batch_pyg = eval_graph_batch_pyg.to(self.device)
                elif isinstance(raw_batch_data, torch.Tensor):
                    eval_real_images_gan_norm = raw_batch_data.to(self.device)

                if eval_real_images_gan_norm is None:
                    print(f"Warning: Could not extract real images for eval batch in {data_split} for arch {self.model_architecture}. Skipping batch.")
                    continue
                current_batch_size = eval_real_images_gan_norm.size(0)
                if current_batch_size == 0: continue

                # --- Prepare Superpixel Conditioning Tensors for G and D (if enabled) ---
                eval_spatial_map_g, eval_spatial_map_d, eval_z_superpixel_g = None, None, None
                g_spatial_active_eval = getattr(self.config.model, f"{self.model_architecture}_g_spatial_cond", False)
                d_spatial_active_eval = getattr(self.config.model, f"{self.model_architecture}_d_spatial_cond", False)
                g_latent_active_eval = self.sp_latent_encoder is not None and \
                                  getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False)

                if self.config.model.use_superpixel_conditioning and eval_segments_map is not None and \
                   (g_spatial_active_eval or d_spatial_active_eval or g_latent_active_eval):
                    eval_real_images_01 = denormalize_image(eval_real_images_gan_norm)
                    if g_spatial_active_eval:
                        eval_spatial_map_g = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_g,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                        if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan", "dcgan"] and \
                           hasattr(self.config.model, "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                           eval_spatial_map_g is not None and eval_spatial_map_g.shape[-1] != 4: # Assuming G starts at 4x4
                             eval_spatial_map_g = F.interpolate(eval_spatial_map_g, size=(4,4), mode='nearest')
                    if d_spatial_active_eval: # For D, use map from real image's segments
                        eval_spatial_map_d = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                    if g_latent_active_eval:
                        mean_sp_feats_eval = calculate_mean_superpixel_features(
                            eval_real_images_01, eval_segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(self.device)
                        eval_z_superpixel_g = self.sp_latent_encoder(mean_sp_feats_eval)

                # --- Architecture-specific evaluation logic ---
                d_real_logits = self.D(eval_real_images_gan_norm, spatial_map_d=eval_spatial_map_d)

                # Generate fake images
                z_noise = torch.randn(current_batch_size, self.config.model.get(f"{self.model_architecture}_z_dim", self.config.model.z_dim), device=self.device)
                g_args_eval = [z_noise]
                g_kwargs_eval = {}
                if self.model_architecture == "gan5_gcn":
                    g_args_eval.extend([eval_real_images_gan_norm, eval_segments_map, eval_adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    z_graph_eval = self.E(eval_graph_batch_pyg) if self.E else torch.zeros(...) # Handle no E
                    if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E: z_graph_eval = torch.zeros_like(z_graph_eval)
                    g_args_eval = [z_graph_eval, current_batch_size]
                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    g_kwargs_eval['spatial_map_g'] = eval_spatial_map_g
                    g_kwargs_eval['z_superpixel_g'] = eval_z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]:
                        g_kwargs_eval['style_mix_prob'] = 0 # No style mixing during eval for consistency
                        # Truncation for StyleGANs during eval
                        if self.config.model.stylegan2_use_truncation and hasattr(self, 'w_avg') and self.w_avg is not None:
                            g_kwargs_eval['truncation_psi'] = self.config.model.stylegan2_truncation_psi_eval
                            g_kwargs_eval['w_avg'] = self.w_avg
                            g_kwargs_eval['truncation_cutoff'] = self.config.model.get('stylegan2_truncation_cutoff_eval', None)
                            # input_is_w might need to be True if G expects w directly for truncation
                    # Add StyleGAN3 specific eval args if any (e.g. related to Fourier features or phase)

                fake_images = self.G(*g_args_eval, **g_kwargs_eval)
                d_fake_logits = self.D(fake_images, spatial_map_d=eval_spatial_map_d) # Pass spatial map if D conditioned

                # Calculate losses
                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD_adv_batch = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits) # Assuming similar structure
                    lossG_batch = self.loss_fn_g_stylegan2(d_fake_logits) # Use d_fake_logits from current fake images
                else: # dcgan, gan5, gan6
                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g(d_fake_logits)

                if self.model_architecture == "projected_gan":
                    # For ProjectedGAN, G loss also includes feature matching
                    # This part is complex as it involves feature_extractor and normalization
                    # Re-using the G loss calculation from training loop for consistency (without G backprop)
                    real_01_eval = denormalize_image(eval_real_images_gan_norm)
                    fake_01_eval = denormalize_image(fake_images)
                    # real_im_norm_fe_eval = self.imagenet_norm(real_01_eval)
                    # fake_im_norm_fe_eval = self.imagenet_norm(fake_01_eval)
                    # Simplified: assume feature_extractor handles this or it's added
                    real_feats_eval = self.feature_extractor(real_01_eval) # real_im_norm_fe_eval
                    fake_feats_eval = self.feature_extractor(fake_01_eval) # fake_im_norm_fe_eval
                    lossG_feat_match_batch = self.loss_fn_g_feat_match(fake_feats_eval['layer4_out'], real_feats_eval['layer4_out'].detach()) # Example layer
                    lossG_batch += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat_match_batch
                    total_g_feat_match_loss += lossG_feat_match_batch.item()


                lossD_batch = lossD_adv_batch # R1 penalty not added in eval
                d_real_logits_mean_batch = d_real_logits.mean()
                d_fake_logits_mean_batch = d_fake_logits.mean()

                total_d_loss += lossD_batch.item()
                total_g_loss += lossG_batch.item()
                total_d_loss_adv += lossD_adv_batch.item()
                total_d_real_logits += d_real_logits_mean_batch.item()
                total_d_fake_logits += d_fake_logits_mean_batch.item()
                num_batches += 1

        self.G.train(); self.D.train() # Set back to train mode
        if self.E: self.E.train()
        if self.sp_latent_encoder: self.sp_latent_encoder.train()

        if num_batches == 0: return {}

        avg_metrics = {
            f"{data_split}/Loss_D": total_d_loss / num_batches,
            f"{data_split}/Loss_G": total_g_loss / num_batches,
            f"{data_split}/Loss_D_Adv": total_d_loss_adv / num_batches,
            f"{data_split}/D_Real_Logits_Mean": total_d_real_logits / num_batches,
            f"{data_split}/D_Fake_Logits_Mean": total_d_fake_logits / num_batches,
        }
        if self.model_architecture == "projected_gan" and total_g_feat_match_loss > 0 : # Only add if computed
            avg_metrics[f"{data_split}/Loss_G_FeatMatch"] = total_g_feat_match_loss / num_batches

        print(f"Evaluation results for {data_split}: {avg_metrics}")
        return avg_metrics

    # Placeholder for checkpointing
    def save_checkpoint(self, epoch, is_best=False):
        # state = {
        #     'epoch': epoch,
        #     'iteration': self.current_iteration,
        #     'config': self.config,
        #     'G_state_dict': self.G.state_dict(),
        #     'D_state_dict': self.D.state_dict(),
        #     'optimizer_G_state_dict': self.optimizer_G.state_dict(),
        #     'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        # }
        # if self.E: state['E_state_dict'] = self.E.state_dict()
        # if self.sp_latent_encoder: state['sp_latent_encoder_state_dict'] = self.sp_latent_encoder.state_dict()
        # if self.w_avg is not None: state['w_avg'] = self.w_avg

        # filename = f"checkpoint_epoch_{epoch}.pth.tar"
        # if self.config.checkpoint_dir:
        #    os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        #    filepath = os.path.join(self.config.checkpoint_dir, filename)
        #    torch.save(state, filepath)
        #    if is_best:
        #        shutil.copyfile(filepath, os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))
        print(f"Checkpoint saved for epoch {epoch} (placeholder).")

    def load_checkpoint(self, checkpoint_path):
        # if not os.path.exists(checkpoint_path):
        #     print(f"Checkpoint path {checkpoint_path} does not exist. Starting from scratch.")
        #     return
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self.G.load_state_dict(checkpoint['G_state_dict'])
        # self.D.load_state_dict(checkpoint['D_state_dict'])
        # self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        # self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        # if self.E and 'E_state_dict' in checkpoint: self.E.load_state_dict(checkpoint['E_state_dict'])
        # if self.sp_latent_encoder and 'sp_latent_encoder_state_dict' in checkpoint:
        #     self.sp_latent_encoder.load_state_dict(checkpoint['sp_latent_encoder_state_dict'])
        # if 'w_avg' in checkpoint and hasattr(self, 'w_avg'): self.w_avg = checkpoint['w_avg']

        # self.current_epoch = checkpoint.get('epoch', 0)
        # self.current_iteration = checkpoint.get('iteration', 0)
        # print(f"Loaded checkpoint from {checkpoint_path}. Resuming from epoch {self.current_epoch}, iteration {self.current_iteration}.")
        print(f"Checkpoint loading from {checkpoint_path} (placeholder).")

# Need to import OmegaConf if used in wandb.init:
from omegaconf import OmegaConf
# Potentially os and shutil for checkpointing
import os
import shutil
