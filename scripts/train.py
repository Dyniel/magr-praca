import argparse
from omegaconf import OmegaConf

# Import the new trainer factory
from src.trainers import get_trainer
from configs.base_config import BaseConfig


def main():
    parser = argparse.ArgumentParser(description="Train a GAN model using a modular trainer architecture.")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to a YAML configuration file to override base defaults.")

    args, unknown_args = parser.parse_known_args()

    # Start with the structured default config
    conf = OmegaConf.structured(BaseConfig)

    # Load from YAML if provided
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

    # --- Trainer Initialization and Training ---
    try:
        # Use the factory to get the correct trainer based on the config
        trainer_instance = get_trainer(config=conf)
        print(f"Trainer for architecture '{conf.model.architecture}' initialized. Starting training process...")
        trainer_instance.train()
    except Exception as e:
        print(f"An critical error occurred during trainer initialization or the training process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()