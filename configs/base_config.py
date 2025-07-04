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
    device: str = "cuda" # "cuda" or "cpu"
    seed: int = 42 # For reproducibility

    # --- Data and Preprocessing ---
    image_size: int = 256
    num_superpixels: int = 150 # Number of superpixels (S)
    slic_compactness: float = 10.0
    # Limit number of images for debugging (0 = use all)
    debug_num_images: int = 0

    # --- Model Hyperparameters ---
    # Shared
    z_dim: int = 256

    # Generator (G)
    g_channels: int = 128 # Base channels for G's GCN blocks
    g_num_gcn_blocks: int = 8
    g_dropout_rate: float = 0.2
    g_ada_in: bool = False # Whether GCNBlocks use AdaIN
    g_spectral_norm: bool = True # Spectral norm for G's WSConv2d layers
    g_final_norm: str = "instancenorm" # 'instancenorm', 'layernorm', or 'none'

    # Discriminator (D)
    d_channels: int = 64 # Base channels for D
    d_spectral_norm: bool = True # Spectral norm for D's WSConv2d and Linear layers
    # d_num_downsampling_layers: int = 2 # Implicitly defined by current D structure

    # --- Training Hyperparameters ---
    batch_size: int = 16
    num_epochs: int = 200
    g_lr: float = 5e-5
    d_lr: float = 2e-4
    beta1: float = 0.0 # Adam optimizer beta1
    beta2: float = 0.99 # Adam optimizer beta2
    r1_gamma: float = 5.0 # R1 gradient penalty weight for Discriminator
    # d_steps_per_g_step: int = 2 # Number of D updates per G update (from gan5)
    # Let's rename for clarity:
    d_updates_per_g_update: int = 2


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
    resume_checkpoint_path: str = None # Path to a .pth.tar checkpoint file to resume training

    # --- Output Directory Management ---
    # This will be the actual directory for this specific run's outputs
    output_dir_run: str = field(init=False)


    def __post_init__(self):
        # Set dynamic paths and dependent configs
        if not self.wandb_project_name: # if not overridden by OmegaConf
            self.wandb_project_name = self.project_name
        if not self.wandb_run_name:
            self.wandb_run_name = self.run_name

        # Construct the full output directory for this run
        self.output_dir_run = os.path.join(self.output_dir_base, self.project_name, self.run_name)

        # The SuperpixelDataset cache_dir is constructed within the dataset itself
        # based on num_superpixels and image_size to allow multiple caches.
        # self.cache_dir = os.path.join(self.cache_dir_base, f"sp_{self.num_superpixels}_is_{self.image_size}")


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
