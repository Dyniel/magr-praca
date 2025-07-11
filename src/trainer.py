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
    SuperpixelLatentEncoder # CycleGANGenerator, CycleGANDiscriminator removed from imports
)
from src.augmentations import ADAManager # Import ADAManager
from src.data_loader import get_dataloader
from src.utils import (
    denormalize_image, generate_spatial_superpixel_map, calculate_mean_superpixel_features,
    toggle_grad, compute_grad_penalty  # R1 gradient penalty
)
from src.losses import HistogramLoss # Import HistogramLoss


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
        self.ada_manager = None # Initialize to None

        # Initialize models, optimizers, loss functions based on self.config (which is now BaseConfig instance)
        self._init_models() # This will also init self.ada_manager if applicable
        self._init_optimizers()
        self._init_loss_functions()

        # For ProjectedGAN feature matching
        if self.model_architecture == "projected_gan":
            self.feature_extractor = FeatureExtractor(
                model_name=self.config.model.projectedgan_feature_extractor_model, # Corrected: was projectedgan_feature_extractor_name
                layers_to_extract=self.config.model.projectedgan_feature_layers_to_extract, # Corrected: was projectedgan_feature_extractor_layers
                pretrained=True,
                requires_grad=False
            ).to(self.device).eval()
            # self.imagenet_norm = ...

        # Initialize ADAManager if StyleGAN2 and ADA is configured
        if self.model_architecture == "stylegan2" and \
           hasattr(self.config.model, 'stylegan2_ada_target_metric_val'): # Check for a key ADA param
            print("Initializing ADAManager for StyleGAN2.")
            self.ada_manager = ADAManager(self.config.model, self.device)


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
        # Removed gan5_gcn and gan6_gat_cnn cases
        if self.model_architecture == "dcgan":
            self.G = DCGANGenerator(self.config).to(self.device)
            self.D = DCGANDiscriminator(self.config).to(self.device)
            self.E = None
        elif self.model_architecture == "stylegan2":
            self.G = StyleGAN2Generator(self.config).to(self.device)
            self.D = StyleGAN2Discriminator(self.config).to(self.device)
            self.E = None
            self.w_avg = None
            if self.config.model.stylegan2_use_truncation:
                pass
        elif self.model_architecture == "stylegan3":
            self.G = StyleGAN3Generator(self.config).to(self.device)
            self.D = StyleGAN3Discriminator(self.config).to(self.device)
            self.E = None
        elif self.model_architecture == "projected_gan":
            self.G = ProjectedGANGenerator(self.config).to(self.device)
            self.D = ProjectedGANDiscriminator(self.config).to(self.device)
            self.E = None

        elif self.model_architecture == "histogan": # HistoGAN uses StyleGAN2 backbone
            self.G = StyleGAN2Generator(self.config).to(self.device)
            self.D = StyleGAN2Discriminator(self.config).to(self.device)
            self.E = None # No separate encoder for HistoGAN in this context
            self.w_avg = None # For StyleGAN2 truncation if used
            if self.config.model.stylegan2_use_truncation: # HistoGAN might use truncation
                pass # Placeholder for w_avg calculation/loading if needed

        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

        self.sp_latent_encoder = None
        if self.config.model.use_superpixel_conditioning and \
                getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False):
            self.sp_latent_encoder = SuperpixelLatentEncoder(
                input_feature_dim=self.config.model.superpixel_feature_dim,

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
        if self.E: g_params += list(self.E.parameters()) # For gan6 E (now removed) or other future E
        if self.sp_latent_encoder: g_params += list(self.sp_latent_encoder.parameters())

        # Removed CycleGAN specific optimizer initialization
        self.optimizer_G = optim.Adam(
            g_params,
            lr=self.config.optimizer.g_lr,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
        )
        self.optimizer_D = optim.Adam(
            self.D.parameters(), # Assumes self.D is the primary discriminator
            lr=self.config.optimizer.d_lr,
            betas=(self.config.optimizer.beta1, self.config.optimizer.beta2)
        )
        print("Optimizers initialized.")

    def _init_loss_functions(self):
        self.r1_gamma = self.config.r1_gamma

        if self.model_architecture in ["gan5_gcn", "gan6_gat_cnn", "dcgan"]:
            self.loss_fn_g_adv = lambda d_fake_logits: F.binary_cross_entropy_with_logits(d_fake_logits, torch.ones_like(d_fake_logits))
            self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
                F.binary_cross_entropy_with_logits(d_real_logits, torch.ones_like(d_real_logits)) + \
                F.binary_cross_entropy_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
        elif self.model_architecture == "stylegan2":
            self.loss_fn_g_adv = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
        elif self.model_architecture == "stylegan3":
            self.loss_fn_g_adv = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
        elif self.model_architecture == "projected_gan":
            self.loss_fn_g_adv = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
                 F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
            self.loss_fn_g_feat_match = nn.MSELoss() # Specific to ProjectedGAN G loss

        elif self.model_architecture == "histogan":
            # HistoGAN uses StyleGAN2's adversarial losses + its own histogram loss
            self.loss_fn_g_adv = lambda d_fake_logits: F.softplus(-d_fake_logits).mean()
            self.loss_fn_d_adv = lambda d_real_logits, d_fake_logits: \
                F.softplus(d_fake_logits).mean() + F.softplus(-d_real_logits).mean()
            self.loss_fn_histogram = HistogramLoss(
                bins=self.config.model.histogan_histogram_bins,
                loss_type=self.config.model.histogan_histogram_loss_type,
                value_range=self.config.model.histogan_image_value_range
            ).to(self.device)
            print(f"HistoGAN Histogram Loss initialized: bins={self.config.model.histogan_histogram_bins}, type={self.config.model.histogan_histogram_loss_type}")

        else:
            self.loss_fn_g_adv = None # Generic name for primary G adversarial loss
            self.loss_fn_d_adv = None # Generic name for primary D adversarial loss

        print(f"Loss functions initialized. R1 Gamma set to: {self.r1_gamma if hasattr(self, 'r1_gamma') else 'N/A (not used by all models)'}")
        if self.model_architecture == "cyclegan":
            print(f"CycleGAN Lambdas: Cycle A/B={self.config.model.cyclegan_lambda_cycle_a}/{self.config.model.cyclegan_lambda_cycle_b}, Identity={self.config.model.cyclegan_lambda_identity}")


    def train(self):
        print(f"Starting training for {self.config.num_epochs} epochs...")


        train_dataloader = get_dataloader(self.config, data_split="train", shuffle=True, drop_last=True)
        if train_dataloader is None:
            print("No training dataloader found. Exiting.")
            return

        for epoch in range(self.current_epoch, self.config.num_epochs):


            self.current_epoch = epoch
            self.G.train()
            self.D.train()
            if self.E: self.E.train()
            if self.sp_latent_encoder: self.sp_latent_encoder.train()

            batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for batch_idx, raw_batch_data in enumerate(batch_iterator):
                if raw_batch_data is None:
                    print(f"Warning: Trainer received a None batch from dataloader at training iteration {self.current_iteration} (epoch {epoch+1}, batch_idx {batch_idx}). Skipping batch.")
                    self.current_iteration +=1
                    continue

                self.current_iteration += 1
                logs = {} # Initialize logs dict for each iteration

                # Removed CycleGAN specific training block.
                # The code below is the original training loop for other GAN architectures.
                real_images_gan_norm = None; segments_map = None; adj_matrix = None; graph_batch_pyg = None

                # Data loading for non-CycleGAN architectures
                if self.model_architecture == "gan6_gat_cnn" and isinstance(raw_batch_data, list) and len(raw_batch_data) > 0: # Workaround
                    real_images_gan_norm = raw_batch_data[0].to(self.device)
                    graph_batch_pyg = None
                elif isinstance(raw_batch_data, dict) and "image" in raw_batch_data: # SuperpixelDataset or ImageDataset
                    real_images_gan_norm = raw_batch_data["image"].to(self.device)
                    if "segments" in raw_batch_data: segments_map = raw_batch_data["segments"].to(self.device)
                    if "adj" in raw_batch_data: adj_matrix = raw_batch_data["adj"].to(self.device)
                elif isinstance(raw_batch_data, tuple) and len(raw_batch_data) == 2: # ImageToGraphDataset
                    real_images_gan_norm, graph_batch_pyg = raw_batch_data
                    real_images_gan_norm = real_images_gan_norm.to(self.device)
                    graph_batch_pyg = graph_batch_pyg.to(self.device)
                elif isinstance(raw_batch_data, torch.Tensor): # Fallback if ImageDataset returns just a tensor
                    real_images_gan_norm = raw_batch_data.to(self.device)

                if real_images_gan_norm is None:
                    print(f"Warning: Could not extract real images for training batch (arch: {self.model_architecture}, type: {type(raw_batch_data)}). Skipping.")
                    continue
                current_batch_size = real_images_gan_norm.size(0)
                if current_batch_size == 0: continue

                # Superpixel conditioning data prep (for non-CycleGAN)
                spatial_map_g, spatial_map_d, z_superpixel_g = None, None, None
                g_spatial_active = getattr(self.config.model, f"{self.model_architecture}_g_spatial_cond", False)
                d_spatial_active = getattr(self.config.model, f"{self.model_architecture}_d_spatial_cond", False)
                g_latent_active = self.sp_latent_encoder is not None and \
                                  getattr(self.config.model, f"{self.model_architecture}_g_latent_cond", False)

                if self.config.model.use_superpixel_conditioning and segments_map is not None and \
                   (g_spatial_active or d_spatial_active or g_latent_active):
                    real_images_01 = denormalize_image(real_images_gan_norm)

                    if g_spatial_active:
                        spatial_map_g = generate_spatial_superpixel_map(
                            segments_map, self.config.model.superpixel_spatial_map_channels_g,
                            self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)
                        if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan", "dcgan"] and \
                           hasattr(self.config.model, "superpixel_spatial_map_channels_g") and self.config.model.superpixel_spatial_map_channels_g > 0 and \
                           spatial_map_g is not None and spatial_map_g.shape[-1] != 4:
                             spatial_map_g = F.interpolate(spatial_map_g, size=(4,4), mode='nearest')

                    if d_spatial_active:
                        spatial_map_d = generate_spatial_superpixel_map(
                        segments_map, self.config.model.superpixel_spatial_map_channels_d,
                        self.config.image_size, self.config.num_superpixels, real_images_01).to(self.device)

                if g_latent_active: # This if block was also incorrectly indented
                    mean_sp_feats = calculate_mean_superpixel_features(
                        real_images_01, segments_map,
                        self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(
                        self.device)
                    z_superpixel_g = self.sp_latent_encoder(mean_sp_feats)

                toggle_grad(self.D, True)
                self.optimizer_D.zero_grad()

                real_images_gan_norm.requires_grad = (self.r1_gamma > 0)

                # Apply ADA if StyleGAN2 and ADA manager is active
                d_input_real_images = real_images_gan_norm
                if self.model_architecture == "stylegan2" and self.ada_manager:
                    d_input_real_images = self.ada_manager.apply_augmentations(real_images_gan_norm)

                if self.model_architecture == "gan6_gat_cnn":
                    d_real_logits = self.D(d_input_real_images) # gan6 D doesn't take spatial_map_d
                else:
                    # For StyleGAN2, d_input_real_images might be augmented
                    d_real_logits = self.D(d_input_real_images, spatial_map_d=spatial_map_d)


                if self.model_architecture == "gan6_gat_cnn":
                    z_dim_to_use = getattr(self.config.model, "gan6_z_dim_noise", self.config.model.z_dim)
                else:
                    z_dim_to_use = getattr(self.config.model, f"{self.model_architecture}_z_dim", self.config.model.z_dim)

                z_noise = torch.randn(current_batch_size, z_dim_to_use, device=self.device)

                g_args = [z_noise]
                g_kwargs = {}
                if self.model_architecture == "gan5_gcn":
                    g_args.extend([real_images_gan_norm, segments_map, adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    if graph_batch_pyg is not None and self.E is not None:
                        z_graph = self.E(graph_batch_pyg)
                        if self.config.model.gan6_gat_cnn_use_null_graph_embedding:
                            print("INFO: gan6_gat_cnn - Using null graph embedding by config.")
                            z_graph = torch.zeros_like(z_graph)
                    elif self.E is not None:
                        print("INFO: gan6_gat_cnn - graph_batch_pyg is None (due to workaround), creating zero z_graph for G's input (D_train step).")
                        z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                    else:
                        print("INFO: gan6_gat_cnn - self.E is None (gan6_use_graph_encoder=False), creating zero z_graph for G's input (D_train step).")
                        z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                    g_args = [z_graph, current_batch_size]
                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    g_kwargs['spatial_map_g'] = spatial_map_g
                    g_kwargs['z_superpixel_g'] = z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]: # Corrected: was model.config...
                        g_kwargs['style_mix_prob'] = getattr(self.config.model, 'stylegan2_style_mix_prob', 0.9)

                        g_kwargs['truncation_psi'] = None

                with torch.no_grad():
                    fake_images = self.G(*g_args, **g_kwargs)


                if self.model_architecture == "gan6_gat_cnn":
                    d_fake_logits = self.D(fake_images.detach())
                else:
                    d_fake_logits = self.D(fake_images.detach(), spatial_map_d=spatial_map_d)

                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits)
                else:
                    lossD = self.loss_fn_d(d_real_logits, d_fake_logits)
                logs["Loss_D_Adv"] = lossD.item()

                if self.r1_gamma > 0:
                    r1_penalty = compute_grad_penalty(d_real_logits, real_images_gan_norm) * self.r1_gamma / 2
                    lossD += r1_penalty
                    logs["Loss_D_R1"] = r1_penalty.item()

                lossD.backward()
                self.optimizer_D.step()
                logs["Loss_D_Total"] = lossD.item()
                toggle_grad(self.D, False)


                if self.current_iteration % self.config.d_updates_per_g_update == 0:
                    toggle_grad(self.G, True)
                    if self.E: toggle_grad(self.E, True)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, True)
                    self.optimizer_G.zero_grad()

                    if self.model_architecture == "gan6_gat_cnn":
                        z_dim_to_use_g = getattr(self.config.model, "gan6_z_dim_noise", self.config.model.z_dim)
                    else:
                        z_dim_to_use_g = getattr(self.config.model, f"{self.model_architecture}_z_dim", self.config.model.z_dim)
                    z_noise_g = torch.randn(current_batch_size, z_dim_to_use_g, device=self.device)

                    g_args_g = [z_noise_g]
                    g_kwargs_g = {}
                    if self.model_architecture == "gan5_gcn":
                        g_args_g.extend([real_images_gan_norm, segments_map, adj_matrix])
                    elif self.model_architecture == "gan6_gat_cnn":
                        if graph_batch_pyg is not None and self.E is not None:
                            z_graph_g = self.E(graph_batch_pyg)
                            if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E:
                                print("INFO: gan6_gat_cnn - Using null graph embedding by config (G_train step).")
                                z_graph_g = torch.zeros_like(z_graph_g)
                        elif self.E is not None:
                            print("INFO: gan6_gat_cnn - graph_batch_pyg is None (due to workaround), creating zero z_graph_g for G's input (G_train step).")
                            z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                            z_graph_g = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                        else:
                            print("INFO: gan6_gat_cnn - self.E is None (gan6_use_graph_encoder=False), creating zero z_graph_g for G's input (G_train step).")

                            z_graph_dim = self.config.model.gan6_z_dim_graph_encoder_output
                            z_graph_g = torch.zeros(current_batch_size, z_graph_dim, device=self.device)
                        g_args_g = [z_graph_g, current_batch_size]
                    elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                        g_kwargs_g['spatial_map_g'] = spatial_map_g
                        g_kwargs_g['z_superpixel_g'] = z_superpixel_g
                        if self.model_architecture in ["stylegan2", "projected_gan"]: # Corrected: was model.config...
                             g_kwargs_g['style_mix_prob'] = getattr(self.config.model, 'stylegan2_style_mix_prob', 0.9)


                    fake_images_for_g = self.G(*g_args_g, **g_kwargs_g)
                    if self.model_architecture == "gan6_gat_cnn":
                        d_fake_logits_for_g = self.D(fake_images_for_g)
                    else:
                        d_fake_logits_for_g = self.D(fake_images_for_g, spatial_map_d=spatial_map_d)

                    if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                        lossG_adv = self.loss_fn_g_stylegan2(d_fake_logits_for_g)
                    else:
                        lossG_adv = self.loss_fn_g(d_fake_logits_for_g)
                    logs["Loss_G_Adv"] = lossG_adv.item()
                    lossG = lossG_adv

                    if self.model_architecture == "histogan":
                        # Ensure images are in the range expected by HistogramLoss (e.g. [-1,1] or [0,1])
                        # The HistogramLoss class itself handles internal normalization to [0,1] based on its value_range config
                        # fake_images_for_g should be in the G's output range (e.g. Tanh -> [-1,1])
                        # real_images_gan_norm is also in G's output range.
                        lossG_hist = self.loss_fn_histogram(fake_images_for_g, real_images_gan_norm)
                        lossG += self.config.model.histogan_histogram_loss_weight * lossG_hist
                        logs["Loss_G_Hist"] = lossG_hist.item()

                    if self.model_architecture == "projected_gan":
                        real_01 = denormalize_image(real_images_gan_norm)
                        fake_01 = denormalize_image(fake_images_for_g)
                        real_feats_dict = self.feature_extractor(real_01)
                        fake_feats_dict = self.feature_extractor(fake_01)

                        lossG_feat = 0.0
                        for key in real_feats_dict:
                            lossG_feat += self.loss_fn_g_feat_match(fake_feats_dict[key], real_feats_dict[key].detach())
                        lossG += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat
                        logs["Loss_G_FeatMatch"] = lossG_feat.item() if isinstance(lossG_feat, torch.Tensor) else lossG_feat

                    lossG.backward()
                    self.optimizer_G.step()
                    logs["Loss_G_Total"] = lossG.item()

                    toggle_grad(self.G, False)
                    if self.E: toggle_grad(self.E, False)
                    if self.sp_latent_encoder: toggle_grad(self.sp_latent_encoder, False)


                if hasattr(self.config, 'logging') and self.current_iteration % self.config.logging.log_freq_step == 0:
                    batch_iterator.set_postfix(logs)
                    if self.config.logging.use_wandb and wandb.run:
                        wandb.log(logs, step=self.current_iteration)
                elif self.current_iteration % getattr(self.config, 'log_freq_step', 100) == 0: # Fallback for older config
                    batch_iterator.set_postfix(logs)
                    if getattr(self.config, 'use_wandb', False) and wandb.run:
                        wandb.log(logs, step=self.current_iteration)

                # ADA p_aug update logic
                if self.model_architecture == "stylegan2" and self.ada_manager and \
                   hasattr(self.config.model, 'stylegan2_ada_interval_kimg') and self.config.model.stylegan2_ada_interval_kimg > 0:

                    # Calculate current kimg (thousands of images processed)
                    # This assumes self.current_iteration is total steps and current_batch_size is constant.
                    # A more robust way might be to sum images seen.
                    # For simplicity:
                    current_kimg = (self.current_iteration * self.config.batch_size) // 1000

                    if self.current_iteration > 0 and \
                       (self.current_iteration * self.config.batch_size) % \
                       (self.config.model.stylegan2_ada_interval_kimg * 1000) < self.config.batch_size:
                        # This condition tries to trigger once when current_kimg crosses an interval boundary.
                        # It's a bit rough due to batch sizes. A dedicated kimg counter would be better.

                        metric_for_ada = 0.0
                        if self.ada_manager.ada_metric_mode == "rt":
                            # Use the mean of real logits from the last D step as a proxy for r_t
                            # This is a simplification. True r_t = E[sign(D(real_aug))].
                            # We are using E[D(real_aug)] directly.
                            metric_for_ada = d_real_logits.mean().item()
                        elif self.ada_manager.ada_metric_mode == "fid":
                            # FID calculation is expensive and usually not done this frequently.
                            # This mode would require FID to be calculated here or fetched if done by another process.
                            # For this implementation, we'll assume FID is not calculated here frequently enough.
                            # So, "rt" mode is more practical with current structure.
                            print("Warning: ADA metric mode 'fid' selected, but frequent FID calculation for ADA update is not implemented here. p_aug will not be updated based on FID.")
                            pass # p_aug won't be updated if FID isn't available

                        if self.ada_manager.ada_metric_mode == "rt": # Only update if metric was available
                            self.ada_manager.update_p_aug(metric_for_ada, current_kimg)
                            if self.config.logging.use_wandb and wandb.run:
                                self.ada_manager.log_status(wandb.log) # Log p_aug to wandb
                                wandb.log({"ada/rt_metric_val": metric_for_ada}, step=self.current_iteration)

                        print(f"ADA Update at iteration {self.current_iteration} (kimg ~{current_kimg}): p_aug = {self.ada_manager.get_p_aug():.4f}, metric_val ({self.ada_manager.ada_metric_mode}) = {metric_for_ada:.4f}")


                # Logging samples, validation, and checkpoints
                # This logic seems a bit duplicated, consolidating for clarity
                log_samples_this_iter = False
                if hasattr(self.config, 'logging'):
                    if self.config.logging.sample_freq_epoch > 0 and \
                       self.current_iteration > 0 and \
                       (self.current_iteration * self.config.batch_size) % \
                       (self.config.logging.sample_freq_epoch * len(train_dataloader.dataset)) < self.config.batch_size:
                        log_samples_this_iter = True
                elif getattr(self.config, 'sample_freq_epoch', 0) > 0: # Fallback
                     if self.current_iteration > 0 and \
                       (self.current_iteration * self.config.batch_size) % \
                       (getattr(self.config, 'sample_freq_epoch', 1) * len(train_dataloader.dataset)) < self.config.batch_size:
                        log_samples_this_iter = True

                if log_samples_this_iter:
                    if (hasattr(self.config, 'logging') and self.config.logging.use_wandb and wandb.run) or \
                       (getattr(self.config, 'use_wandb', False) and wandb.run):
                        eval_metrics = self._evaluate_on_split("val") # Also logs samples within _evaluate_on_split
                        if eval_metrics: # eval_metrics can be {} if dataloader is None
                            wandb.log(eval_metrics, step=self.current_iteration)

                # Checkpointing (end of epoch based)
                is_last_batch_of_epoch = (batch_idx == len(train_dataloader) - 1)
                checkpoint_freq = getattr(self.config.logging, 'checkpoint_freq_epoch', getattr(self.config, 'checkpoint_freq_epoch', 10))
                if is_last_batch_of_epoch and (epoch + 1) % checkpoint_freq == 0:
                    self.save_checkpoint(epoch=self.current_epoch, is_best=False)


            print(f"Epoch {epoch+1} completed.")


        print("Training finished.")
        if hasattr(self.config, 'logging') and self.config.logging.use_wandb:
            wandb.finish()
        elif getattr(self.config, 'use_wandb', False):
            wandb.finish()

    def _evaluate_on_split(self, data_split: str):
        print(f"Evaluating on {data_split} split...")
        eval_dataloader = get_dataloader(self.config, data_split=data_split, shuffle=False, drop_last=False)

        if eval_dataloader is None:
            print(f"No dataloader for {data_split} split (path likely not configured). Skipping evaluation.")
            return {}

        self.G.eval()
        self.D.eval()
        if self.E: self.E.eval()
        if self.sp_latent_encoder: self.sp_latent_encoder.eval()

        total_d_loss = 0.0; total_g_loss = 0.0; total_d_loss_adv = 0.0
        total_d_real_logits = 0.0; total_d_fake_logits = 0.0
        total_g_feat_match_loss = 0.0

        num_batches = 0

        with torch.no_grad():
            for batch_idx, raw_batch_data in enumerate(tqdm(eval_dataloader, desc=f"Evaluating {data_split}")):
                if raw_batch_data is None:
                    print(
                        f"Warning: Trainer received a None batch from dataloader during {data_split} evaluation (batch_idx {batch_idx}). Skipping batch.")
                    continue

                lossD_batch = torch.tensor(0.0, device=self.device); lossG_batch = torch.tensor(0.0, device=self.device)
                lossD_adv_batch = torch.tensor(0.0, device=self.device); d_real_logits_mean_batch = torch.tensor(0.0, device=self.device)
                d_fake_logits_mean_batch = torch.tensor(0.0, device=self.device); lossG_feat_match_batch = torch.tensor(0.0, device=self.device)
                current_batch_size = 0

                eval_real_images_gan_norm = None; eval_segments_map = None; eval_adj_matrix = None; eval_graph_batch_pyg = None

                if self.model_architecture == "gan6_gat_cnn" and isinstance(raw_batch_data, list) and len(raw_batch_data) > 0:
                    print(f"INFO: Applying workaround for list-type batch in trainer (eval {data_split}, gan6_gat_cnn). Batch idx: {batch_idx}")
                    eval_real_images_gan_norm = raw_batch_data[0].to(self.device)
                    eval_graph_batch_pyg = None
                elif isinstance(raw_batch_data, dict) and "image" in raw_batch_data:
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
                           eval_spatial_map_g is not None and eval_spatial_map_g.shape[-1] != 4:
                             eval_spatial_map_g = F.interpolate(eval_spatial_map_g, size=(4,4), mode='nearest')
                    if d_spatial_active_eval:
                        eval_spatial_map_d = generate_spatial_superpixel_map(
                            eval_segments_map, self.config.model.superpixel_spatial_map_channels_d,
                            self.config.image_size, self.config.num_superpixels, eval_real_images_01).to(self.device)
                    if g_latent_active_eval:
                        mean_sp_feats_eval = calculate_mean_superpixel_features(
                            eval_real_images_01, eval_segments_map,
                            self.sp_latent_encoder.num_superpixels, self.config.model.superpixel_feature_dim).to(
                            self.device)
                        eval_z_superpixel_g = self.sp_latent_encoder(mean_sp_feats_eval)


                if self.model_architecture == "gan6_gat_cnn":
                    d_real_logits = self.D(eval_real_images_gan_norm)
                else:
                    d_real_logits = self.D(eval_real_images_gan_norm, spatial_map_d=eval_spatial_map_d)

                if self.model_architecture == "gan6_gat_cnn":
                    z_dim_to_use_eval = getattr(self.config.model, "gan6_z_dim_noise", self.config.model.z_dim)
                else:
                    z_dim_to_use_eval = getattr(self.config.model, f"{self.model_architecture}_z_dim", self.config.model.z_dim)
                z_noise = torch.randn(current_batch_size, z_dim_to_use_eval, device=self.device)


                g_args_eval = [z_noise]
                g_kwargs_eval = {}
                if self.model_architecture == "gan5_gcn":
                    g_args_eval.extend([eval_real_images_gan_norm, eval_segments_map, eval_adj_matrix])
                elif self.model_architecture == "gan6_gat_cnn":
                    if eval_graph_batch_pyg is not None and self.E is not None:
                        z_graph_eval = self.E(eval_graph_batch_pyg)
                        if self.config.model.gan6_gat_cnn_use_null_graph_embedding and self.E:
                            z_graph_eval = torch.zeros_like(z_graph_eval)
                    elif self.E is not None:
                        print(f"INFO: gan6_gat_cnn - eval_graph_batch_pyg is None (due to workaround), creating zero z_graph for G's input (eval {data_split} step).")
                        z_graph_dim_eval = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph_eval = torch.zeros(current_batch_size, z_graph_dim_eval, device=self.device)
                    else:
                        print(f"INFO: gan6_gat_cnn - self.E is None (gan6_use_graph_encoder=False), creating zero z_graph for G's input (eval {data_split} step).")
                        z_graph_dim_eval = self.config.model.gan6_z_dim_graph_encoder_output
                        z_graph_eval = torch.zeros(current_batch_size, z_graph_dim_eval, device=self.device)

                    g_args_eval = [z_graph_eval, current_batch_size]

                elif self.model_architecture in ["dcgan", "stylegan2", "stylegan3", "projected_gan"]:
                    g_kwargs_eval['spatial_map_g'] = eval_spatial_map_g
                    g_kwargs_eval['z_superpixel_g'] = eval_z_superpixel_g
                    if self.model_architecture in ["stylegan2", "projected_gan"]:
                        g_kwargs_eval['style_mix_prob'] = 0
                        if self.config.model.stylegan2_use_truncation and hasattr(self, 'w_avg') and self.w_avg is not None:
                            g_kwargs_eval['truncation_psi'] = self.config.model.stylegan2_truncation_psi_eval
                            g_kwargs_eval['w_avg'] = self.w_avg
                            g_kwargs_eval['truncation_cutoff'] = getattr(self.config.model, 'stylegan2_truncation_cutoff_eval', None)

                fake_images = self.G(*g_args_eval, **g_kwargs_eval)


                if self.model_architecture == "gan6_gat_cnn":
                    d_fake_logits = self.D(fake_images)
                else:
                    d_fake_logits = self.D(fake_images, spatial_map_d=eval_spatial_map_d)

                if self.model_architecture in ["stylegan2", "stylegan3", "projected_gan"]:
                    lossD_adv_batch = self.loss_fn_d_stylegan2(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g_stylegan2(d_fake_logits)
                else:
                    lossD_adv_batch = self.loss_fn_d(d_real_logits, d_fake_logits)
                    lossG_batch = self.loss_fn_g(d_fake_logits)

                if self.model_architecture == "projected_gan":
                    real_01_eval = denormalize_image(eval_real_images_gan_norm)
                    fake_01_eval = denormalize_image(fake_images)
                    real_feats_eval = self.feature_extractor(real_01_eval)
                    fake_feats_eval = self.feature_extractor(fake_01_eval)
                    lossG_feat_match_batch = self.loss_fn_g_feat_match(fake_feats_eval['layer4_out'], real_feats_eval['layer4_out'].detach())
                    lossG_batch += self.config.model.projectedgan_feature_matching_loss_weight * lossG_feat_match_batch
                    total_g_feat_match_loss += lossG_feat_match_batch.item()


                d_real_logits_mean_batch = d_real_logits.mean()
                d_fake_logits_mean_batch = d_fake_logits.mean()

                total_d_loss += lossD_batch.item()
                total_g_loss += lossG_batch.item()
                total_d_loss_adv += lossD_adv_batch.item()
                total_d_real_logits += d_real_logits_mean_batch.item()
                total_d_fake_logits += d_fake_logits_mean_batch.item()
                num_batches += 1

        self.G.train(); self.D.train()

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
        if self.model_architecture == "projected_gan" and total_g_feat_match_loss > 0 :
            avg_metrics[f"{data_split}/Loss_G_FeatMatch"] = total_g_feat_match_loss / num_batches

        print(f"Evaluation results for {data_split}: {avg_metrics}")
        return avg_metrics

    def save_checkpoint(self, epoch, is_best=False):
        print(f"Checkpoint saved for epoch {epoch} (placeholder).")

    def load_checkpoint(self, checkpoint_path):
        print(f"Checkpoint loading from {checkpoint_path} (placeholder).")


from omegaconf import OmegaConf
import os
import shutil