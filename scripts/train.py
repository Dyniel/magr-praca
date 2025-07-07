import argparse
from omegaconf import OmegaConf

# Import the refactored Trainer from src module
from src.trainer import Trainer
# Assuming BaseConfig is used for structuring the config and lives in configs.base_config
from configs.base_config import BaseConfig


def main():
    parser = argparse.ArgumentParser(description="Train a GAN model using the main Trainer from src.trainer.")
    parser.add_argument("--config_file", type=str, default=None,
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

    # Apply command-line overrides
    if unknown_args:
        try:
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
    try:
        actual_config_object = OmegaConf.to_object(conf)
    except Exception as e:
        print(f"Error converting OmegaConf to BaseConfig object: {e}")
        print("The Trainer will receive the OmegaConf object directly. Ensure it's compatible.")
        actual_config_object = conf

        # --- Trainer Initialization and Training ---
    try:
        trainer_instance = Trainer(config=actual_config_object)
        print("Trainer from 'src.trainer' initialized. Starting training process...")
        trainer_instance.train()
    except Exception as e:
        print(f"An critical error occurred during trainer initialization or the training process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()