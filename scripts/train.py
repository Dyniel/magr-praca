import argparse
import sys
import os

# Add project root to Python path to allow direct import of src, configs
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from omegaconf import OmegaConf

from configs.base_config import BaseConfig
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train a Superpixel-Conditioned GAN")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/experiment_config.yaml",
        help="Path to the experiment configuration YAML file."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint file (.pth.tar) to resume training from. Overrides 'resume_checkpoint_path' in config file."
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config parameters (e.g., batch_size=32 num_epochs=100)."
    )

    args = parser.parse_args()

    # 1. Start with the structured base configuration
    config = OmegaConf.structured(BaseConfig)

    # 2. Load configuration from the specified YAML file
    if os.path.exists(args.config_file):
        yaml_config = OmegaConf.load(args.config_file)
        config = OmegaConf.merge(config, yaml_config)
        print(f"Loaded configuration from: {args.config_file}")
    else:
        print(f"Warning: Config file {args.config_file} not found. Using default base configuration.")

    # 3. Merge overrides from command line arguments
    # Example: python scripts/train.py --config_file configs/my_exp.yaml batch_size=8 num_epochs=50
    if args.overrides:
        cli_overrides = OmegaConf.from_dotlist(args.overrides)
        config = OmegaConf.merge(config, cli_overrides)
        print(f"Applied CLI overrides: {args.overrides}")

    # If --resume_from is provided via CLI, it takes precedence
    if args.resume_from:
        config.resume_checkpoint_path = args.resume_from
        print(f"CLI override: Resuming from checkpoint: {config.resume_checkpoint_path}")

    # OmegaConf's __post_init__ for BaseConfig should have run if BaseConfig was used in OmegaConf.structured
    # However, OmegaConf sometimes requires explicit re-instantiation or specific handling for post_init with dataclasses.
    # Let's ensure BaseConfig's post_init logic is triggered after all merges.
    # One way is to convert to a dict and back, or use a method if BaseConfig had one.
    # A simpler way for this setup: if BaseConfig's post_init is crucial and not auto-triggered by OmegaConf merge,
    # we might need to call it manually or ensure OmegaConf.to_object(config) is used carefully.
    # For now, assuming OmegaConf handles dataclass __post_init__ correctly upon merge/access.
    # If not, `config = BaseConfig(**OmegaConf.to_container(config, resolve=True))` could be an option,
    # but that loses OmegaConf's features. Best to rely on OmegaConf's behavior or test.
    # The BaseConfig.__post_init__ is designed to work even if called multiple times or if OmegaConf sets field during merge.
    # Let's explicitly call it on the final config object if it's an instance of BaseConfig,
    # or if we convert it. For now, we assume OmegaConf handles it.
    # The BaseConfig.__post_init__ sets derived fields like `wandb_project_name` and `output_dir_run`.
    # If these are not set, it means __post_init__ didn't run as expected after merges.
    # A quick check:
    if not hasattr(config, 'output_dir_run') or config.output_dir_run is None:
        print("Running __post_init__ for config adjustments...")
        # This is a bit of a hack. Ideally, OmegaConf handles this seamlessly.
        # If OmegaConf creates a new object not of type BaseConfig, this won't work.
        # It's better if BaseConfig's post_init can be called idempotently or if
        # OmegaConf provides a hook.
        # For now, let's assume the fields that __post_init__ sets are correctly populated
        # by OmegaConf if they are not part of the YAML/CLI overrides.
        # The current BaseConfig sets them using field(init=False) and updates in __post_init__.
        # This should generally work.
        # Let's add a manual call for safety, but this is usually not needed if structured configs are used well.
        temp_dict = OmegaConf.to_container(config, resolve=True)
        final_config = BaseConfig(**temp_dict) # This will run __post_init__
        config = OmegaConf.structured(final_config) # Convert back to OmegaConf object if needed by trainer

    print("\nFinal Configuration:")
    print(OmegaConf.to_yaml(config))

    # Create and run the trainer
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()

print("scripts/train.py created.")
