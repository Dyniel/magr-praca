import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import wandb  # Assuming wandb is used for logging, can be made optional

# Project specific imports - these might need adjustment based on actual file structure and names
from src.models import (
    Generator as GAN5Generator, Discriminator as GAN5Discriminator,  # gan5_gcn
    GraphEncoderGAT, GeneratorCNN as GAN6Generator, DiscriminatorCNN as GAN6Discriminator,  # gan6_gat_cnn
    DCGANGenerator, DCGANDiscriminator,
    StyleGAN2Generator, StyleGAN2Discriminator,
    StyleGAN3Generator, StyleGAN3Discriminator,  # Assuming these exist
    ProjectedGANGenerator, ProjectedGANDiscriminator, FeatureExtractor,  # Assuming these exist
    SuperpixelLatentEncoder
)
from src.data_loader import get_dataloader
from src.utils import (
    denormalize_image, generate_spatial_superpixel_map, calculate_mean_superpixel_features,
    toggle_grad, compute_grad_penalty  # R1 gradient penalty
)


# Add other necessary loss functions or utilities
# from src.losses import generator_loss_nonsaturating, discriminator_loss_r1, ...


class Trainer:
    def __init__(self, config):  # Parameter name changed back to config
        # Log the OmegaConf object passed as 'config'
        if hasattr(config, 'logging') and config.logging.use_wandb and config.logging.wandb_project_name:
            wandb.init(
                project=config.logging.wandb_project_name,
                entity=config.logging.wandb_entity,
                name=config.logging.wandb_run_name,
                config=OmegaConf.to_container(config, resolve=True)  # Log the original OmegaConf
            )
        elif hasattr(config, 'use_wandb') and config.use_wandb:  # Fallback
            print(
                "Warning: 'logging' attribute not found in config, but 'use_wandb' is true. Attempting legacy WandB init.")
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
            if hasattr(self.config, 'logging'):  # Check if logging config exists
                if hasattr(self, 'G') and self.G is not None:
                    wandb.watch(self.G, log="all", log_freq=self.config.logging.wandb_watch_freq_g)
                if hasattr(self, 'D') and self.D is not None:
                    wandb.watch(self.D, log="all", log_freq=self.config.logging.wandb_watch_freq_d)
            elif hasattr(self.config, 'use_wandb') and self.config.use_wandb:  # Fallback for older config structure
                if hasattr(self, 'G') and self.G is not None: wandb.watch(self.G, log="all")
                if hasattr(self, 'D') and self.D is not None: wandb.watch(self.D, log="all")

    def _init_models(self):
        # Generator and Discriminator
        if self.model_architecture == "gan5_gcn":
            self.G = GAN5Generator(self.config).to(self.device)
            self.D = GAN5Discriminator(self.config).to(self.device)
            self.E = None  # gan5_gcn doesn't use a separate E model in this structure
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
                pass  # This would be calculated later or loaded.
        elif self.model_architecture == "stylegan3":
            self.G = StyleGAN3Generator(self.config).to(self.device)
            self.D = StyleGAN3Discriminator(self.config).to(self.device)
            self.E = None
        elif self.model_architecture == "projected_gan":
            self.G = ProjectedGANGenerator(self.config).to(self.device)  # Based on StyleGAN2
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
                input_feature_dim=self.config.model.superpixel_feature_dim,  # e.g., 3 for RGB
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
            lr=self.config.optimizer.g_lr,  # Updated path
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)  # Updated path
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=self.config.optimizer.d_lr,  # Updated path
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)  # Updated path

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
            self.loss_fn_g = lambda d_fake_logits: F.binary_cross_entropy_with_logits(d_fake_logits,
                                                                                      torch.ones_like(d_fake_logits))
            self.loss_fn_d = lambda d_real_logits, d_fake_logits: \
                F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits)) + \
                F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            # self.r1_gamma is already set from top-level config
        elif self.model_architecture == "stylegan2":
            self.loss_fn_g_stylegan2 = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()  # Non-saturating
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
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()  # Or hinge
            self.loss_fn_g_feat_match = nn.MSELoss()
            # self.r1_gamma from top-level
        else:
            self.loss_fn_g = None
            self.loss_fn_d = None
            # self.r1_gamma will use the value from top-level config, or its default if not overridden by sweep.

        print(f"Loss functions initialized. R1 Gamma set to: {self.r1_gamma}")

    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs...")  # Use self.config.num_epochs

        train_dataloader = get_dataloader(self.config, data_split="train", shuffle=True, drop_last=True)
        if train_dataloader is None:
            print("No training dataloader found. Exiting.")
            return

        for epoch in range(self.current_epoch, self.config.num_epochs):  # Use self.config.num_epochs

            self.current_epoch = epoch
            self.G.train()
            self.D.train()
            if self.E: self.E.train()
            if self.sp_latent_encoder: self.sp_latent_encoder.train()

            # Corrected: self.config.num_epochs instead of self.config.training.epochs
            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
            for batch_idx, raw_batch_data in enumerate(batch_iterator):
                if raw_batch_data is None:
                    print(
                        f"Warning: Trainer received a None batch from dataloader at training iteration {self.current_iteration} (epoch {epoch + 1}, batch_idx {batch_idx}). Skipping batch.")
                    self.current_iteration += 1  # Still increment iteration if you want to count skipped batches
                    continue

                self.current_iteration += 1
                logs = {}

                # --- Prepare inputs for current batch (real images, segments, etc.) ---
                real_images_gan_norm = None;
                segments_map = None;
                adj_matrix = None;
                graph_batch_pyg = None

                # Workaround for gan6_gat_cnn if dataloader returns a list [images_tensor]
                if self.model_architecture == "gan6_gat_cnn" and isinstance(raw_batch_data, list) and len(
                        raw_batch_data) > 0:
                    print(
                        f"INFO: Applying workaround for list-type batch in trainer (gan6_gat_cnn). Batch idx: {batch_idx}")
                    real_images_gan_norm = raw_batch_data[0].to(self.device)
                    graph_batch_pyg = None  # Explicitly set to None, as graph data is missing from the list
                    # The rest of the GAN6 logic must be able to handle graph_batch_pyg being None
                    # or E(None) if E is called with it.
                elif isinstance(raw_batch_data, dict) and "image" in raw_batch_data:  # For SuperpixelDataset (gan5)
                    real_images_gan_norm = raw_batch_data["image"].to(self.device)
                    if "segments" in raw_batch_data: segments_map = raw_batch_data["segments"].to(self.device)
                    if "adj" in raw_batch_data: adj_matrix = raw_batch_data["adj"].to(self.device)
                elif isinstance(raw_batch_data, tuple) and len(
                        raw_batch_data) == 2:  # For graph-based GANs like gan6 (expected path)
                    real_images_gan_norm, graph_batch_pyg = raw_batch_data
                    real_images_gan_norm = real_images_gan_norm.to(self.device)
                    graph_batch_pyg = graph_batch_pyg.to(self.device)  # This should be a PyGBatch object
                elif isinstance(raw_batch_data,
                                torch.Tensor):  # Simple image tensor (e.g. for DCGAN, StyleGANs without graph inputs)
                    real_images_gan_norm = raw_batch_data.to(self.device)

                if real_images_gan_norm is None:
                    print(
                        f"Warning: Could not extract real images for training batch (arch: {self.model_architecture}, type: {type(raw_batch_data)}). Skipping.")
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
                    real_images_01 = denormalize_image(real_images_gan_norm)  # For feature/map generation
                    if g_spatial_active:
                        spatial_map_g = generate_spatial_superpixel_map(
                            segments_map, self.config.model.superpixel_spatial_map_channels_g,
                            self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                        # Resize if G expects a fixed small map (e.g. StyleGANs starting at 4x4)
                        if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan", "dcgan"] and \
                                hasattr(self.config.model,
                                        "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                                spatial_map_g is not None and spatial_map_g.shape[-1] != 4:  # Assuming G starts at 4x4
                            spatial_map_g = F.interpolate(spatial_map_g, size=(4, 4), mode='nearest')

                    if d_spatial_active:  # For D, use map from real image's segments
                        spatial_map_d = generate_spatial_superpixel_map(
                            segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                        # D usually takes full-res spatial map, no resize needed here unless specific D arch.
                    if g_latent_active:
                        mean_sp_feats = calculate_mean_superpixel_features(
                            real_images_01, segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(
                            self.device)
                        z_superpixel_g = self.sp_latent_encoder(mean_sp_feats)

                # --- Train Discriminator ---
                toggle_grad(self.D, True)
                self.optimizer_D.zero_grad()

                # Real images
                real_images_gan_norm.requires_grad = (self.r1_gamma > 0)  # For R1 penalty
                if self.model_architecture == "gan6_gat_cnn":
                    d_real_logits = self.D(real_images_gan_norm)  # DiscriminatorCNN does not take spatial_map_d
                else:
                    d_real_logits = self.D(real_images_gan_norm, spatial_map_d=spatial_map_d)

                # Fake images
                # Use getattr for z_dim as self.config.model is a dataclass
                z_dim_attribute_name = f"{self.model_architecture}_z_dim"
                # For gan6_gat_cnn, the noise dim is specifically gan6_z_dim_noise
                if self.model_architecture == "gan6_gat_cnn":
                    z_dim_attribute_name = "gan6_z_dim_noise"

                # Fallback to the general z_dim if the specific one (like gan6_z_dim_noise) isn't found,
                # or if the constructed attribute name (like gan6_gat_cnn_z_dim) isn't found.
                # The most specific one (e.g. gan6_z_dim_noise) should be tried first if applicable.
                if hasattr(self.config.model, z_dim_attribute_name):
                    z_dim_to_use = getattr(self.config.model, z_dim_attribute_name)
                else:  # Fallback to general z_dim
                    z_dim_to_use = self.config.model.z_dim

                z_noise = torch.randn(current_batch_size, z_dim_to_use, device=self.device)

                # Generator forward pass arguments vary by architecture
                g_args = [z_noise]
                g_kwargs = {}
                if self.model_architecture == "gan5_gcn":
                    g_args.extend([real_images_gan_norm, segments_map, adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    if graph_batch_pyg is not None and self.E is not None:
                        z_graph = self.E(graph_batch_pyg)
                        # This handles gan6_gat_cnn_use_null_graph_embedding if true
                        if self.config.model.gan6_gat_cnn_use_null_graph_embedding:
                            print("INFO: gan6_gat_cnn - Using null graph embedding by config.")
                            z_graph = torch.zeros_like(z_graph)
                    elif self.E is not None:  # graph_batch_pyg is None due to workaround, but E exists
                        print(
                            "INFO: gan6_gat_cnn - graph_batch_pyg is None (due to workaround), creating zero z_graph for G's input (D_train step).")
                        z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                    else:  # No E (self.config.model.gan6_use_graph_encoder is False)
                        # GAN6Generator still expects a z_graph argument. Provide a zero tensor.
                        print(
                            "INFO: gan6_gat_cnn - self.E is None (gan6_use_graph_encoder=False), creating zero z_graph for G's input (D_train step).")
                        z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                    g_args = [z_graph, current_batch_size]  # GAN6 G takes z_graph and batch_size
                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    # These models can take additional conditioning arguments
                    g_kwargs['spatial_map_g'] = spatial_map_g
                    g_kwargs['z_superpixel_g'] = z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]:
                        g_kwargs['style_mix_prob'] = self.config.model.get('stylegan2_style_mix_prob', 0.9)
                        g_kwargs['truncation_psi'] = None  # No truncation during training typically
                    # Add other specific args if needed, like input_is_w for StyleGANs if doing w-space training

                with torch.no_grad():  # Detach G's output for D training
                    fake_images = self.G(*g_args, **g_kwargs)
                if self.model_architecture == "gan6_gat_cnn":
                    d_fake_logits = self.D(fake_images.detach())  # DiscriminatorCNN does not take spatial_map_d
                else:
                    d_fake_logits = self.D(fake_images.detach(),
                                           spatial_map_d=spatial_map_d)  # Pass spatial_map_d also for fake if D is conditioned

                # Discriminator loss
                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits)  # Assuming similar loss structure
                else:  # dcgan, gan5, gan6
                    lossD = self.loss_fn_d(d_real_logits, d_fake_logits)
                logs["Loss_D_Adv"] = lossD.item()

                # R1 gradient penalty (common for StyleGANs, can be added to others)
                if self.r1_gamma > 0:
                    r1_penalty = compute_grad_penalty(d_real_logits, real_images_gan_norm) * self.r1_gamma / 2
                    lossD += r1_penalty
                    logs["Loss_D_R1"] = r1_penalty.item()

                lossD.backward()
                self.optimizer_D.step()
                logs["Loss_D_Total"] = lossD.item()
                toggle_grad(self.D, False)  # Freeze D for G training

                # --- Train Generator ---
                # Potentially less frequent G updates
                # Corrected: self.config.d_updates_per_g_update instead of self.config.training.g_steps_per_d_step
                if self.current_iteration % self.config.d_updates_per_g_update == 0:
                    toggle_grad(self.G, True)
                    if self.E: toggle_grad(self.E, True)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, True)
                    self.optimizer_G.zero_grad()

                    # Regenerate fake images with G grads enabled
                    # Noise and G args should be same as above for consistency in this step, or resampled
                    # Use getattr for z_dim as self.config.model is a dataclass
                    z_dim_attribute_name_g = f"{self.model_architecture}_z_dim"
                    if self.model_architecture == "gan6_gat_cnn":  # For gan6_gat_cnn, noise dim is specifically gan6_z_dim_noise
                        z_dim_attribute_name_g = "gan6_z_dim_noise"

                    if hasattr(self.config.model, z_dim_attribute_name_g):
                        z_dim_to_use_g = getattr(self.config.model, z_dim_attribute_name_g)
                    else:  # Fallback to general z_dim
                        z_dim_to_use_g = self.config.model.z_dim

                    z_noise_g = torch.randn(current_batch_size, z_dim_to_use_g, device=self.device)
                    g_args_g = [z_noise_g]
                    g_kwargs_g = {}  # Reset kwargs for G training pass
                    if self.model_architecture == "gan5_gcn":
                        g_args_g.extend(
                            [real_images_gan_norm, segments_map, adj_matrix])  # Use same conditioning as D saw
                    elif self.model_architecture == "gan6_gat_cnn":
                        if graph_batch_pyg is not None and self.E is not None:
                            z_graph_g = self.E(graph_batch_pyg)
                            # This handles gan6_gat_cnn_use_null_graph_embedding if true
                            if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E:
                                print("INFO: gan6_gat_cnn - Using null graph embedding by config (G_train step).")
                                z_graph_g = torch.zeros_like(z_graph_g)
                        elif self.E is not None:  # graph_batch_pyg is None due to workaround, but E exists
                            print(
                                "INFO: gan6_gat_cnn - graph_batch_pyg is None (due to workaround), creating zero z_graph_g for G's input (G_train step).")
                            z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                            z_graph_g = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                        else:  # No E (self.config.model.gan6_use_graph_encoder is False)
                            # GAN6Generator still expects a z_graph argument. Provide a zero tensor.
                            print(
                                "INFO: gan6_gat_cnn - self.E is None (gan6_use_graph_encoder=False), creating zero z_graph_g for G's input (G_train step).")
                            z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                            z_graph_g = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                        g_args_g = [z_graph_g, current_batch_size]
                    elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                        g_kwargs_g['spatial_map_g'] = spatial_map_g
                        g_kwargs_g['z_superpixel_g'] = z_superpixel_g  # Use same z_superpixel_g derived from reals
                        if self.model_architecture in ["stylegan2", "projected_gan"]:
                            g_kwargs_g['style_mix_prob'] = self.config.model.get('stylegan2_style_mix_prob', 0.9)

                    fake_images_for_g = self.G(*g_args_g, **g_kwargs_g)
                    if self.model_architecture == "gan6_gat_cnn":
                        d_fake_logits_for_g = self.D(fake_images_for_g)  # DiscriminatorCNN does not take spatial_map_d
                    else:
                        d_fake_logits_for_g = self.D(fake_images_for_g,
                                                     spatial_map_d=spatial_map_d)  # Pass spatial_map_d if D conditioned

                    if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                        lossG_adv = self.loss_fn_g_stylegan2(d_fake_logits_for_g)
                    else:
                        lossG_adv = self.loss_fn_g(d_fake_logits_for_g)
                    logs["Loss_G_Adv"] = lossG_adv.item()
                    lossG = lossG_adv

                    # ProjectedGAN: Feature Matching Loss
                    if self.model_architecture == "projected_gan":
                        # Normalize real and fake images for feature extractor
                        # Assuming denormalize_image brings to [0,1] range
                        real_01 = denormalize_image(real_images_gan_norm)
                        fake_01 = denormalize_image(fake_images_for_g)
                        # Apply ImageNet normalization (or whatever FeatureExtractor expects)
                        # This needs self.imagenet_norm to be defined (e.g., torchvision.transforms.Normalize)
                        # real_im_norm_fe = self.imagenet_norm(real_01)
                        # fake_im_norm_fe = self.imagenet_norm(fake_01)
                        # For now, assuming feature_extractor can handle [0,1] or [-1,1] directly, or this is added:
                        # This is a simplification. Proper normalization is crucial.
                        real_feats_dict = self.feature_extractor(real_01)  # Pass real_im_norm_fe
                        fake_feats_dict = self.feature_extractor(fake_01)  # Pass fake_im_norm_fe

                        lossG_feat = 0.0
                        for key in real_feats_dict:  # Iterate over features from different layers
                            lossG_feat += self.loss_fn_g_feat_match(fake_feats_dict[key], real_feats_dict[key].detach())
                        lossG += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat
                        logs["Loss_G_FeatMatch"] = lossG_feat.item() if isinstance(lossG_feat,
                                                                                   torch.Tensor) else lossG_feat

                    lossG.backward()
                    self.optimizer_G.step()
                    logs["Loss_G_Total"] = lossG.item()

                    toggle_grad(self.G, False)
                    if self.E: toggle_grad(self.E, False)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, False)

                # Logging
                # Corrected: Check for logging attribute and use self.config.logging.*
                if hasattr(self.config, 'logging') and self.current_iteration % self.config.logging.log_freq_step == 0:
                    batch_iterator.set_postfix(logs)
                    if self.config.logging.use_wandb:
                        wandb.log(logs, step=self.current_iteration)
                elif self.current_iteration % getattr(self.config, 'log_freq_step', 100) == 0:  # Fallback
                    batch_iterator.set_postfix(logs)
                    if getattr(self.config, 'use_wandb', False):
                        wandb.log(logs, step=self.current_iteration)

                # Evaluation and Checkpointing
                # Corrected: Check for logging attribute and use self.config.logging.*
                if hasattr(self.config,
                           'logging') and self.current_iteration % self.config.logging.sample_freq_epoch == 0:
                    eval_metrics = self._evaluate_on_split("val")
                    if self.config.logging.use_wandb and eval_metrics:
                        wandb.log(eval_metrics, step=self.current_iteration)

                    if epoch % self.config.logging.checkpoint_freq_epoch == 0 and batch_idx == len(
                            train_dataloader) - 1:
                        self.save_checkpoint(epoch=self.current_epoch, is_best=False)
                elif self.current_iteration % getattr(self.config, 'sample_freq_epoch', 1) == 0:  # Fallback
                    eval_metrics = self._evaluate_on_split("val")
                    if getattr(self.config, 'use_wandb', False) and eval_metrics:
                        wandb.log(eval_metrics, step=self.current_iteration)
                    if epoch % getattr(self.config, 'checkpoint_freq_epoch', 10) == 0 and batch_idx == len(
                            train_dataloader) - 1:
                        self.save_checkpoint(epoch=self.current_epoch, is_best=False)

            # End of epoch
            print(f"Epoch {epoch + 1} completed.")

        print("Training finished.")
        # Corrected: Check for logging attribute and use self.config.logging.use_wandb
        if hasattr(self.config, 'logging') and self.config.logging.use_wandb:
            wandb.finish()
        elif getattr(self.config, 'use_wandb', False):  # Fallback
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
        total_g_feat_match_loss = 0.0  # For ProjectedGAN
        num_batches = 0

        with torch.no_grad():
            for batch_idx, raw_batch_data in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {data_split}")):
                if raw_batch_data is None:
                    print(
                        f"Warning: Trainer received a None batch from dataloader during {data_split} evaluation (batch_idx {batch_idx}). Skipping batch.")
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
                eval_real_images_gan_norm = None;
                eval_segments_map = None;
                eval_adj_matrix = None;
                eval_graph_batch_pyg = None
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
                    print(
                        f"Warning: Could not extract real images for eval batch in {data_split} for arch {self.model_architecture}. Skipping batch.")
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
                                hasattr(self.config.model,
                                        "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                                eval_spatial_map_g is not None and eval_spatial_map_g.shape[
                            -1] != 4:  # Assuming G starts at 4x4
                            eval_spatial_map_g = F.interpolate(eval_spatial_map_g, size=(4, 4), mode='nearest')
                    if d_spatial_active_eval:  # For D, use map from real image's segments
                        eval_spatial_map_d = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                    if g_latent_active_eval:
                        mean_sp_feats_eval = calculate_mean_superpixel_features(
                            eval_real_images_01, eval_segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(
                            self.device)
                        eval_z_superpixel_g = self.sp_latent_encoder(mean_sp_feats_eval)

                # --- Architecture-specific evaluation logic ---
                d_real_logits = self.D(eval_real_images_gan_norm, spatial_map_d=eval_spatial_map_d)

                # Generate fake images
                z_noise = torch.randn(current_batch_size, self.config.model.get(f"{self.model_architecture}_z_dim",
                                                                                self.config.model.z_dim),
                                      device=self.device)
                g_args_eval = [z_noise]
                g_kwargs_eval = {}
                if self.model_architecture == "gan5_gcn":
                    g_args_eval.extend([eval_real_images_gan_norm, eval_segments_map, eval_adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    z_graph_eval = self.E(eval_graph_batch_pyg) if self.E else torch.zeros(...)  # Handle no E
                    if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E: z_graph_eval = torch.zeros_like(
                        z_graph_eval)
                    g_args_eval = [z_graph_eval, current_batch_size]
                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    g_kwargs_eval['spatial_map_g'] = eval_spatial_map_g
                    g_kwargs_eval['z_superpixel_g'] = eval_z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]:
                        g_kwargs_eval['style_mix_prob'] = 0  # No style mixing during eval for consistency
                        # Truncation for StyleGANs during eval
                        if self.config.model.stylegan2_use_truncation and hasattr(self,
                                                                                  'w_avg') and self.w_avg is not None:
                            g_kwargs_eval['truncation_psi'] = self.config.model.stylegan2_truncation_psi_eval
                            g_kwargs_eval['w_avg'] = self.w_avg
                            g_kwargs_eval['truncation_cutoff'] = self.config.model.get(
                                'stylegan2_truncation_cutoff_eval', None)
                            # input_is_w might need to be True if G expects w directly for truncation
                    # Add StyleGAN3 specific eval args if any (e.g. related to Fourier features or phase)

                fake_images = self.G(*g_args_eval, **g_kwargs_eval)
                d_fake_logits = self.D(fake_images,
                                       spatial_map_d=eval_spatial_map_d)  # Pass spatial map if D conditioned

                # Calculate losses
                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD_adv_batch = self.loss_fn_d_stylegan2(d_real_logits,
                                                               d_fake_logits)  # Assuming similar structure
                    lossG_batch = self.loss_fn_g_stylegan2(d_fake_logits)  # Use d_fake_logits from current fake images
                else:  # dcgan, gan5, gan6
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
                    real_feats_eval = self.feature_extractor(real_01_eval)  # real_im_norm_fe_eval
                    fake_feats_eval = self.feature_extractor(fake_01_eval)  # fake_im_norm_fe_eval
                    lossG_feat_match_batch = self.loss_fn_g_feat_match(fake_feats_eval['layer4_out'], real_feats_eval[
                        'layer4_out'].detach())  # Example layer
                    lossG_batch += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat_match_batch
                    total_g_feat_match_loss += lossG_feat_match_batch.item()

                lossD_batch = lossD_adv_batch  # R1 penalty not added in eval
                d_real_logits_mean_batch = d_real_logits.mean()
                d_fake_logits_mean_batch = d_fake_logits.mean()

                total_d_loss += lossD_batch.item()
                total_g_loss += lossG_batch.item()
                total_d_loss_adv += lossD_adv_batch.item()
                total_d_real_logits += d_real_logits_mean_batch.item()
                total_d_fake_logits += d_fake_logits_mean_batch.item()
                num_batches += 1

        self.G.train();
        self.D.train()  # Set back to train mode
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
        if self.model_architecture == "projected_gan" and total_g_feat_match_loss > 0:  # Only add if computed
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
