import argparse
from omegaconf import OmegaConf

# Import the refactored Trainer from src module
from src.trainer import Trainer
# Assuming BaseConfig is used for structuring the config and lives in configs.base_config
from configs.base_config import BaseConfig


def main():
    parser = argparse.ArgumentParser(description="Train a GAN model using the main Trainer from src.trainer.")
    # This argument will be parsed by argparse and not passed to OmegaConf
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to a YAML configuration file to override base defaults.")

    # Allow unknown args for OmegaConf to parse as dot-list overrides
    args, unknown_args = parser.parse_known_args()

    # Start with the structured default config from BaseConfig
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

    # Apply command-line overrides from the remaining arguments
    if unknown_args:
        try:
            # The unknown_args should be in a format that from_dotlist can parse,
            # e.g., ['foo.bar=10', 'baz=True']
            print(f"Processing dotlist overrides from: {unknown_args}")
            cli_overrides = OmegaConf.from_dotlist(unknown_args)
            conf = OmegaConf.merge(conf, cli_overrides)
            print(f"Applied CLI overrides: {unknown_args}")
        except Exception as e:
            print(f"Error applying CLI overrides: {e}.")

    print("\nFinal configuration after all merges:")
    try:
        print(OmegaConf.to_yaml(conf))
    except Exception as e:
        print(f"Could not print final config as YAML: {e}")

    # Convert OmegaConf to the actual BaseConfig dataclass instance
    # actual_config_object = OmegaConf.to_object(conf) # Trainer will handle conversion if needed

        # --- Trainer Initialization and Training ---
    try:
        # Pass the OmegaConf object `conf` directly to the Trainer
        trainer_instance = Trainer(config=conf)
        print("Trainer from 'src.trainer' initialized. Starting training process...")
        trainer_instance.train()
    except Exception as e:
        print(f"An critical error occurred during trainer initialization or the training process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()