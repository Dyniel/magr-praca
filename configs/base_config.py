from typing import Optional
from dataclasses import dataclass, field
import os

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    use_wandb: bool = True
    wandb_project_name: Optional[str] = None # Will be derived from BaseConfig.project_name in __post_init__
    wandb_run_name: Optional[str] = None     # Will be derived from BaseConfig.run_name in __post_init__
    wandb_entity: Optional[str] = None       # User's W&B entity name
    wandb_watch_freq_g: int = 1000           # How often to log gradients/parameters for G
    wandb_watch_freq_d: int = 1000           # How often to log gradients/parameters for D
    log_freq_step: int = 100
    image_log_freq: int = 500
    sample_freq_epoch: int = 1
    num_samples_to_log: int = 10
    checkpoint_freq_epoch: int = 10
    calculate_fid: bool = True
    fid_num_images: int = 5000


@dataclass
class OptimizerConfig:
    """Configuration for optimizers."""
    g_lr: float = 5e-5
    d_lr: float = 2e-4
    beta1: float = 0.0 # Adam optimizer beta1 for G and D
    beta2: float = 0.99 # Adam optimizer beta2 for G and D
    lambda_gp: float = 10.0
    # Potentially add separate betas for G and D if needed:
    # g_beta1: float = 0.0
    # g_beta2: float = 0.99
    # d_beta1: float = 0.0
    # d_beta2: float = 0.99


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    architecture: str = "stylegan2"  # Changed default from gan5_gcn
                                     # Options: "dcgan", "stylegan2", "stylegan3", "projected_gan", "cyclegan", "histogan"

    # --- Shared Hyperparameters ---
    # z_dim is now model-specific, e.g., stylegan2_z_dim, dcgan_z_dim, etc.
    # If a truly shared z_dim is needed later, it can be re-added.
    # For now, each model defines its own latent dimension(s).

    # --- Parameters for DCGAN architecture ---
    dcgan_z_dim: int = 100 # Typical z_dim for DCGAN
    dcgan_g_feat: int = 64 # Feature map size for DCGAN Generator
    dcgan_d_feat: int = 64 # Feature map size for DCGAN Discriminator

    # --- Parameters for StyleGAN2 architecture ---
    stylegan2_z_dim: int = 512
    stylegan2_w_dim: int = 512
    stylegan2_n_mlp: int = 8 # Number of layers in mapping network
    stylegan2_lr_mul_mapping: float = 0.01 # Learning rate multiplier for mapping network
    stylegan2_channel_multiplier: int = 1 # Channel multiplier for G and D resolutions
    stylegan2_style_mix_prob: float = 0.9  # Probability of applying style mixing.
    stylegan2_use_truncation: bool = True  # Whether to use truncation trick during inference/sampling.
    stylegan2_truncation_psi: float = 0.7  # Truncation psi for training/sampling (if not overridden for eval).
    stylegan2_truncation_cutoff: Optional[int] = None  # Number of layers to apply truncation to (all if None or 0).
    stylegan2_truncation_psi_eval: float = 0.7 # Truncation psi specifically for evaluation.
    stylegan2_truncation_cutoff_eval: Optional[int] = None # Truncation cutoff specifically for evaluation.
    # stylegan2_blur_kernel: list[int] = field(default_factory=lambda: [1,3,3,1]) # Blur kernel for FIR filtering
    # stylegan2_g_reg_every: int = 4 # How often to perform G path regularization (if implemented)
    # stylegan2_d_reg_every: int = 16 # How often to perform D R1 regularization

    # StyleGAN2-ADA specific parameters
    stylegan2_ada_target_metric_val: float = 0.6 # Target value for the chosen ADA metric (e.g., r_v, FID threshold)
    stylegan2_ada_interval_kimg: float = 4.0     # How often to update p_aug (in kimg), changed to float
    stylegan2_ada_kimg_target_ramp_up: int = 500 # Duration over which to ramp up p_aug towards initial_p_aug_target if metric is too low
    stylegan2_ada_p_aug_initial: float = 0.0 # Initial augmentation probability
    stylegan2_ada_p_aug_step: float = 0.005   # Step size for adjusting p_aug
    stylegan2_ada_augment_pipeline: list[str] = field(default_factory=lambda: [
        "brightness", "contrast", "lumaflip", "hue", "saturation", # color
        "imgcrop", "geom", # geom
        # "cutout" # Often separate
    ])
    # Individual augmentation probabilities (can be overridden in YAML)
    # These are the 'xflip', 'rotate90', 'xint', 'xint_max' etc. from StyleGAN2-ADA paper.
    # For simplicity here, we'll have a single p_aug and apply the selected pipeline.
    # More granular control could be added later.
    stylegan2_ada_metric_mode: str = "rt" # "rt" (sign of D output), "fid" (if FID is frequent enough)

    # --- Parameters for StyleGAN3 architecture (simplified) ---
    stylegan3_z_dim: int = 512
    stylegan3_w_dim: int = 512
    stylegan3_n_mlp: int = 8
    stylegan3_lr_mul_mapping: float = 0.01
    stylegan3_channel_multiplier: int = 1
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
    projectedgan_d_channel_multiplier: int = 1
    projectedgan_blur_kernel: list[int] = field(default_factory=lambda: [1,3,3,1])
    projectedgan_feature_extractor_name: str = "resnet50" # e.g., "resnet50", "efficientnet_b0"
    # projectedgan_feature_extractor_path: Optional[str] = None # Optional path to custom weights
    projectedgan_feature_layers_to_extract: Optional[list[str]] = None # Layers from feature extractor, if None, model defaults used
    projectedgan_projection_dims: int = 256 # Example dim if D were to project features (not used in current D model)
    projectedgan_feature_matching_loss_weight: float = 10.0 # Weight for feature matching loss for G


    # --- Parameters for HistoGAN architecture ---
    # HistoGAN typically builds on StyleGAN2. These params are for its specific loss.
    # StyleGAN2 specific parameters (stylegan2_z_dim, etc.) will be used by HistoGAN's base.
    histogan_histogram_loss_weight: float = 1.0
    histogan_histogram_bins: int = 256
    histogan_histogram_loss_type: str = 'l1' # 'l1', 'l2', or 'cosine'
    # Expected input range for images when calculating histogram loss.
    # The HistogramLoss class will normalize images from this range to [0,1] before computing histograms.
    # If your generator outputs [-1,1], set this to [-1.0, 1.0]. If [0,1], set to [0.0, 1.0].
    histogan_image_value_range: tuple[float, float] = field(default_factory=lambda: (-1.0, 1.0))


@dataclass
class BaseConfig:
    # --- System and Paths ---
    project_name: str = "SuperpixelGAN_Refactored"
    # Base directory for all outputs (logs, checkpoints, samples)
    # Auto-set based on project_name if not specified
    output_dir_base: str = "experiment_outputs"
    run_name: str = "default_run" # Specific name for this run, used for subfolder in output_dir
    # Full path to the dataset
    dataset_path: str = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/train"
    dataset_path_val: Optional[str] = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/val"
    dataset_path_test: Optional[str] = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/test"
    # Directory to cache precomputed superpixels
    cache_dir: str = "/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc/cache"
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
    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    # --- Optimizer Configuration ---
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig())

    # --- Training Hyperparameters ---
    batch_size: int = 2
    num_epochs: int = 200
    # g_lr, d_lr, beta1, beta2 are now in OptimizerConfig
    r1_gamma: float = 5.0 # R1 gradient penalty weight for Discriminator. This is sweepable directly.
    # d_steps_per_g_step: int = 2 # Number of D updates per G update (from gan5)
    # Let's rename for clarity:
    d_updates_per_g_update: int = 2
    gradient_accumulation_steps: int = 1 # Number of steps to accumulate gradients before optimizer step

    # --- Logging Configuration ---
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())


    # --- FID Configuration (kept separate for now or could be part of LoggingConfig) ---
    # fid_incep_path: str = "path/to/inception_v3_fid.pt" # Not used if using pytorch-fid's default
    fid_num_images: int = 5000 # Number of real/fake images to use for FID
    fid_batch_size: int = 2   # Batch size for generating images for FID
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
        self.output_dir_run = os.path.join(self.output_dir_base, self.project_name, self.run_name)

        # Set wandb project and run names based on top-level config if not already set in logging object
        # This handles cases where logging fields might not be in the YAML but should be derived.
        # Ensure self.logging is an object with attributes before trying to access/set them.
        # OmegaConf.structured should ensure self.logging is LoggingConfig instance.
        # If it's merged from a dict that doesn't have these, they'd be None from default_factory.
        if isinstance(self.logging, LoggingConfig):
            if self.logging.wandb_project_name is None:
                self.logging.wandb_project_name = self.project_name
            if self.logging.wandb_run_name is None:
                self.logging.wandb_run_name = self.run_name
        else:
            # This case should ideally not be reached if OmegaConf.structured works as expected
            # and YAML files correctly define the logging structure or omit it to use defaults.
            # If self.logging is somehow not a LoggingConfig object, we might need to log a warning
            # or attempt a more robust re-initialization, but that complexity is what
            # we are trying to avoid by relying on OmegaConf's structuring.
            # For now, we assume OmegaConf handles the type correctly.
            pass


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

    # Check if logging is correctly initialized before accessing its attributes
    if bc.logging:
        print(f"WandB Project: {bc.logging.wandb_project_name}")
        print(f"WandB Run: {bc.logging.wandb_run_name}")
    else:
        print("Logging config is not initialized.")


    # Example of how SuperpixelDataset might form its specific cache dir
    sp_cache_example = os.path.join(bc.cache_dir, f"sp_{bc.num_superpixels}_is_{bc.image_size}")
    print(f"Example SuperpixelDataset cache dir: {sp_cache_example}")

print("configs/base_config.py created.")
