from typing import Optional
from dataclasses import dataclass, field
import os

@dataclass
class BaseConfig:
    # --- System and Paths ---
    project_name: str = "SuperpixelGAN_Refactored"
    # Base directory for all outputs (logs, checkpoints, samples)
    # Auto-set based on project_name if not specified
    output_dir_base: str = "experiment_outputs"
    run_name: str = "default_run" # Specific name for this run, used for subfolder in output_dir
    # Full path to the dataset
    dataset_path: str = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc"
    # Directory to cache precomputed superpixels
    cache_dir: str = "superpixel_cache"
    num_workers: int = 4 # For DataLoader
    device: str = "cuda" # "cuda" or "cpu". Trainer will verify availability and fallback to CPU if needed.
    use_cuda: bool = True # Explicit flag, though `device` field is primary. Kept for compatibility if Trainer directly uses it.
    seed: Optional[int] = 42 # For reproducibility, None means no explicit seed setting in main script.

    # --- Data and Preprocessing ---
    image_size: int = 256
    num_superpixels: int = 150 # Number of superpixels (S)
    slic_compactness: float = 10.0
    # Limit number of images for debugging (0 = use all)
    debug_num_images: int = 0

    # --- Model Configuration ---
    # This will be a nested structure, defined below
    model: 'ModelConfig' = field(default_factory=lambda: ModelConfig())

    # --- Optimizer Configuration ---
    optimizer: 'OptimizerConfig' = field(default_factory=lambda: OptimizerConfig())

    # --- Training Hyperparameters ---
    batch_size: int = 16
    num_epochs: int = 200
    # g_lr, d_lr, beta1, beta2 are now in OptimizerConfig
    r1_gamma: float = 5.0 # R1 gradient penalty weight for Discriminator. This is sweepable directly.
    # d_steps_per_g_step: int = 2 # Number of D updates per G update (from gan5)
    # Let's rename for clarity:
    d_updates_per_g_update: int = 2
    gradient_accumulation_steps: int = 1 # Number of steps to accumulate gradients before optimizer step


    # --- Logging and Evaluation ---
    use_wandb: bool = True # Enable Weights & Biases logging
    wandb_project_name: str = field(init=False) # Will be set from project_name
    wandb_run_name: str = field(init=False) # Will be set from run_name
    log_freq_step: int = 100 # Log metrics every N training steps
    sample_freq_epoch: int = 1 # Generate and log sample images every N epochs
    num_samples_to_log: int = 10 # Number of samples for image logging
    checkpoint_freq_epoch: int = 10 # Save checkpoints every N epochs
    # For FID calculation (if implemented later)
    # fid_incep_path: str = "path/to/inception_v3_fid.pt" # Not used if using pytorch-fid's default
    fid_num_images: int = 5000 # Number of real/fake images to use for FID
    fid_batch_size: int = 32   # Batch size for generating images for FID
    fid_freq_epoch: int = 5    # How often (in epochs) to calculate FID. Can be expensive.
    # path_to_real_images_for_fid: str = None # Optional: path to a pre-selected dir of real images for FID consistency. If None, uses current dataset.
    # enable_fid_calculation: bool = False # Set to True in experiment YAML to enable
    enable_fid_calculation: bool = True # Default to True, can be disabled in YAML.

    # --- Resume Training ---
    resume_checkpoint_path: Optional[str] = None # Path to a .pth.tar checkpoint file to resume training

    # --- Output Directory Management ---
    # This will be the actual directory for this specific run's outputs
    output_dir_run: str = field(init=False)

    def __post_init__(self):
        # Set dynamic paths and dependent configs

        # Initialize init=False fields if they haven't been set (e.g., by OmegaConf merge)
        if not hasattr(self, 'wandb_project_name') or getattr(self, 'wandb_project_name', None) is None:
            self.wandb_project_name = self.project_name
        if not hasattr(self, 'wandb_run_name') or getattr(self, 'wandb_run_name', None) is None:
            self.wandb_run_name = self.run_name

        # Construct the full output directory for this run
        self.output_dir_run = os.path.join(self.output_dir_base, self.project_name, self.run_name)

        # Ensure model and optimizer defaults are populated correctly after potential merges
        if not isinstance(self.model, ModelConfig):
            current_model_config_dict = {}
            if self.model is not None:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(self.model):
                    current_model_config_dict = OmegaConf.to_container(self.model, resolve=True)
                elif isinstance(self.model, dict):
                    current_model_config_dict = self.model
            default_model_conf = ModelConfig()
            for key, value in current_model_config_dict.items():
                if hasattr(default_model_conf, key): setattr(default_model_conf, key, value)
            self.model = default_model_conf
        elif self.model is None: self.model = ModelConfig()

        if not isinstance(self.optimizer, OptimizerConfig):
            current_optimizer_config_dict = {}
            if self.optimizer is not None:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(self.optimizer):
                    current_optimizer_config_dict = OmegaConf.to_container(self.optimizer, resolve=True)
                elif isinstance(self.optimizer, dict):
                    current_optimizer_config_dict = self.optimizer
            default_optimizer_conf = OptimizerConfig()
            for key, value in current_optimizer_config_dict.items():
                if hasattr(default_optimizer_conf, key): setattr(default_optimizer_conf, key, value)
            self.optimizer = default_optimizer_conf
        elif self.optimizer is None: self.optimizer = OptimizerConfig()


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    g_lr: float = 5e-5
    d_lr: float = 2e-4
    beta1: float = 0.0 # Adam optimizer beta1 for G and D
    beta2: float = 0.99 # Adam optimizer beta2 for G and D
    # Potentially add separate betas for G and D if needed:
    # g_beta1: float = 0.0
    # g_beta2: float = 0.99
    # d_beta1: float = 0.0
    # d_beta2: float = 0.99


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    architecture: str = "gan5_gcn"  # Options: "gan5_gcn", "gan6_gat_cnn"

    # --- Shared Hyperparameters (used by both architectures if applicable) ---
    z_dim: int = 256 # General latent dimension (e.g., for noise in gan5, or part of combined z in gan6)

    # --- Parameters for gan5_gcn architecture ---
    # (These were previously top-level in BaseConfig)
    # Generator (G for gan5)
    g_channels: int = 128       # Base channels for G's GCN blocks
    g_num_gcn_blocks: int = 8
    g_dropout_rate: float = 0.2
    g_ada_in: bool = False      # Whether GCNBlocks use AdaIN
    g_spectral_norm: bool = True # Spectral norm for G's WSConv2d layers
    g_final_norm: str = "instancenorm" # 'instancenorm', 'layernorm', or 'none'

    # Discriminator (D for gan5)
    d_channels: int = 64        # Base channels for D
    d_spectral_norm: bool = True  # Spectral norm for D's WSConv2d and Linear layers

    # --- Parameters for gan6_gat_cnn architecture ---
    # Graph Encoder (E for gan6)
    gat_dim: int = 128
    gat_heads: int = 4
    gat_layers: int = 3
    gat_dropout: float = 0.0 # Dropout for GAT layers
    gan6_z_dim_graph_encoder_output: int = 128 # Output dim of GraphEncoder (z_graph)

    # Generator (G_cnn for gan6)
    gan6_z_dim_noise: int = 128 # Dimension of z_noise to be combined with z_graph
    gan6_gen_init_size: int = 4
    gan6_gen_feat_start: int = 512
    gan6_gen_spectral_norm: bool = True

    # Discriminator (D_cnn for gan6)
    gan6_d_feat_start: int = 64
    gan6_d_final_conv_size: int = 16 # Spatial size of feature map before FC layer in D
    gan6_d_spectral_norm: bool = True
    gan6_d_spectral_norm_fc: bool = True # Whether to apply SN to the final FC layer of D_cnn

    # Superpixel settings specific to gan6 graph creation (if different from global num_superpixels)
    # These are used by ImageToGraphDataset via config.model.*
    gan6_num_superpixels: int = 200      # Default for gan6, can differ from global num_superpixels
    gan6_slic_compactness: float = 10.0  # Default for gan6

    # --- Parameters for DCGAN architecture ---
    dcgan_g_feat: int = 64 # Feature map size for DCGAN Generator
    dcgan_d_feat: int = 64 # Feature map size for DCGAN Discriminator

    # --- Parameters for StyleGAN2 architecture ---
    stylegan2_z_dim: int = 512
    stylegan2_w_dim: int = 512
    stylegan2_n_mlp: int = 8 # Number of layers in mapping network
    stylegan2_lr_mul_mapping: float = 0.01 # Learning rate multiplier for mapping network
    stylegan2_channel_multiplier: int = 2 # Channel multiplier for G and D resolutions
    # stylegan2_blur_kernel: list[int] = field(default_factory=lambda: [1,3,3,1]) # Blur kernel for FIR filtering
    # stylegan2_g_reg_every: int = 4 # How often to perform G path regularization (if implemented)
    # stylegan2_d_reg_every: int = 16 # How often to perform D R1 regularization

    # --- Parameters for StyleGAN3 architecture (simplified) ---
    stylegan3_z_dim: int = 512
    stylegan3_w_dim: int = 512
    stylegan3_n_mlp: int = 8
    stylegan3_lr_mul_mapping: float = 0.01
    stylegan3_channel_multiplier: int = 2
    stylegan3_fir_kernel: list[int] = field(default_factory=lambda: [1,3,3,1]) # Basic FIR kernel for up/downsampling (simplified)
    # stylegan3_magnitude_ema_beta: float = 0.5 # For magnitude-based EMA in some variants

    # --- Parameters for Projected GAN architecture ---
    # As ProjectedGANGenerator inherits StyleGAN2Generator, it will use stylegan2_* G params from above.
    # We only need D-specific and feature extractor specific params here for ProjectedGAN.

    # --- Superpixel Conditioning Configs (applied per model type if flags are set) ---
    # General flag to enable any superpixel feature processing for a model
    use_superpixel_conditioning: bool = False # Top-level switch, if False, model-specific flags are ignored

    # C1: Spatial Conditioning for Generator (input concat)
    # Number of channels for the spatial superpixel map (e.g., 1 for segment IDs, 3 for mean color map, S for one-hot)
    superpixel_spatial_map_channels_g: int = 1

    # C4: Spatial Conditioning for Discriminator (input concat)
    superpixel_spatial_map_channels_d: int = 1

    # C2: Latent Code Modulation from Superpixel Embedding
    # Parameters for a simple superpixel feature encoder (e.g., MLP on mean colors)
    superpixel_latent_encoder_enabled: bool = False
    superpixel_feature_dim: int = 3 # e.g., RGB mean color
    superpixel_latent_encoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 128])
    superpixel_latent_embedding_dim: int = 128 # Output dim of z_sp

    # Model-specific flags to enable types of conditioning
    # For DCGAN
    dcgan_g_spatial_cond: bool = False # Enable C1 for DCGAN G
    dcgan_g_latent_cond: bool = False  # Enable C2 for DCGAN G
    dcgan_d_spatial_cond: bool = False # Enable C4 for DCGAN D

    # For StyleGAN2
    stylegan2_g_spatial_cond: bool = False # Enable C1 for StyleGAN2 G (e.g. concat to initial const or early layer)
    stylegan2_g_latent_cond: bool = False  # Enable C2 for StyleGAN2 G (e.g. concat to z or modulate w)
    stylegan2_d_spatial_cond: bool = False # Enable C4 for StyleGAN2 D

    # For StyleGAN3 (simplified)
    stylegan3_g_spatial_cond: bool = False # C1 for StyleGAN3 G (e.g. concat to fourier features if made spatial)
    stylegan3_g_latent_cond: bool = False  # C2 for StyleGAN3 G
    stylegan3_d_spatial_cond: bool = False # C4 for StyleGAN3 D

    # For ProjectedGAN (Generator is StyleGAN2 based, Discriminator is custom)
    # G conditioning will be via stylegan2_*_cond flags if ProjectedGAN G uses them.
    # For D, it's specific.
    projectedgan_d_spatial_cond: bool = False # C4 for ProjectedGAN D's own path

    # For gan5_gcn (already superpixel based) - ablation might mean *disabling* parts
    gan5_gcn_disable_gcn_blocks: bool = False # Example ablation for gan5

    # For gan6_gat_cnn (already superpixel based) - ablation might mean zeroing graph embedding
    gan6_gat_cnn_use_null_graph_embedding: bool = False # Example ablation for gan6
    projectedgan_d_channel_multiplier: int = 2
    projectedgan_blur_kernel: list[int] = field(default_factory=lambda: [1,3,3,1])
    projectedgan_feature_extractor_name: str = "resnet50" # e.g., "resnet50", "efficientnet_b0"
    # projectedgan_feature_extractor_path: Optional[str] = None # Optional path to custom weights
    projectedgan_feature_layers_to_extract: Optional[list[str]] = None # Layers from feature extractor, if None, model defaults used
    projectedgan_projection_dims: int = 256 # Example dim if D were to project features (not used in current D model)
    projectedgan_feature_matching_loss_weight: float = 10.0 # Weight for feature matching loss for G


# Example of how to use:
# from omegaconf import OmegaConf
# cfg = OmegaConf.structured(BaseConfig)
# # To load from YAML and merge:
# # cli_conf = OmegaConf.from_cli() # For command line overrides
# # file_conf = OmegaConf.load("experiment_config.yaml")
# # cfg = OmegaConf.merge(cfg, file_conf, cli_conf)

# print(cfg)

if __name__ == "__main__":
    # This is just for testing the BaseConfig standalone
    bc = BaseConfig(run_name="my_test_run_1")
    print(f"Project Name: {bc.project_name}")
    print(f"Dataset Path: {bc.dataset_path}")
    print(f"Cache Directory (base for dataset): {bc.cache_dir}")
    print(f"Output directory for this run: {bc.output_dir_run}")
    print(f"WandB Project: {bc.wandb_project_name}")
    print(f"WandB Run: {bc.wandb_run_name}")

    # Example of how SuperpixelDataset might form its specific cache dir
    sp_cache_example = os.path.join(bc.cache_dir, f"sp_{bc.num_superpixels}_is_{bc.image_size}")
    print(f"Example SuperpixelDataset cache dir: {sp_cache_example}")

print("configs/base_config.py created.")
