import subprocess
import tempfile
import os
import glob
import csv
from omegaconf import OmegaConf, DictConfig

# Assuming BaseConfig is importable and structured appropriately
# Add paths for local imports if necessary, or ensure your PYTHONPATH is set
try:
    from configs.base_config import BaseConfig, ModelConfig
except ModuleNotFoundError:
    # Simple fallback if running script directly and paths aren't set up
    # This might require adjusting if your BaseConfig has complex dependencies at import time
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from configs.base_config import BaseConfig, ModelConfig

def get_experiment_configurations() -> list[dict]:
    """
    Defines a list of experiment configurations to run.
    Each configuration is a dictionary with:
        - name: A descriptive name for the experiment.
        - model_architecture: The architecture to use.
        - run_name_suffix: Suffix for WandB run name.
        - config_overrides: Dictionary of OmegaConf dot-paths and values to override.
    """
    experiments = []
    config_files = glob.glob("configs/example_tests/*.yaml")

    for config_file in config_files:
        conf = OmegaConf.load(config_file)
        model_architecture = conf.model.architecture
        run_name = os.path.basename(config_file).replace(".yaml", "")

        overrides = {
            "model.architecture": model_architecture,
            "run_name": run_name,
            "num_epochs": 200,
            "batch_size": 32,
            "enable_fid_calculation": True,
        }

        experiments.append({
            "name": run_name,
            "model_architecture": model_architecture,
            "config_overrides": overrides
        })

    return experiments

def run_single_experiment(exp_config: dict, base_cfg_obj: OmegaConf):
    """Runs a single experiment using scripts/train.py."""
    print(f"\n{'='*30}\nRunning Experiment: {exp_config['name']}\n{'='*30}")

    # Create a mutable copy of the base config
    conf = base_cfg_obj.copy() # type: ignore

    # Apply overrides
    for key, value in exp_config["config_overrides"].items():
        OmegaConf.update(conf, key, value) # type: ignore

    # Ensure essential settings for the run
    conf.logging.use_wandb = True # Force WandB for these experiments
    # run_name should be set by overrides, but double check
    if "run_name" not in exp_config["config_overrides"]:
        print(f"Warning: 'run_name' not in overrides for {exp_config['name']}. Using default.")
        conf.run_name = exp_config['name'].replace(" ", "_").lower()
    else:
        conf.run_name = exp_config['config_overrides']['run_name']


    # Create a temporary YAML file for this specific configuration
    # tempfile.NamedTemporaryFile creates a file that is deleted when closed.
    # We need to pass the filename to subprocess, so we manage it manually.
    temp_dir = tempfile.mkdtemp()
    temp_config_path = os.path.join(temp_dir, "temp_experiment_config.yaml")

    try:
        OmegaConf.save(config=conf, f=temp_config_path)
        print(f"Saved temporary config for {exp_config['name']} to {temp_config_path}")
        print(f"Final configuration for this run ({conf.run_name}):")
        # print(OmegaConf.to_yaml(conf)) # Can be verbose

        # Construct and run the training command
        # Ensure train.py is executable or called via python -m
        # Using python -m is generally more robust for module resolution
        command = [
            "python", "-m", "scripts.train",
            "--config_file", temp_config_path
        ]
        print(f"Executing command: {' '.join(command)}")

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream output
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')

        process.wait() # Wait for the subprocess to complete

        if process.returncode == 0:
            print(f"Experiment {exp_config['name']} (Run: {conf.run_name}) completed successfully.")
            return True
        else:
            print(f"Error: Experiment {exp_config['name']} (Run: {conf.run_name}) failed with return code {process.returncode}.")
            return False

    except Exception as e:
        print(f"An error occurred while setting up or running experiment {exp_config['name']}: {e}")
        return False
    finally:
        # Clean up the temporary file and directory
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        print(f"Cleaned up temporary config for {exp_config['name']}.")


def main():
    print("Starting batch of GAN experiments...")
    experiments_to_run = get_experiment_configurations()
    print(f"Found {len(experiments_to_run)} experiment configurations to run.")

    # Load the structured base configuration once
    # This ensures that any defaults or type hints from BaseConfig are respected
    # before overrides are applied.
    base_omega_conf = OmegaConf.structured(BaseConfig)

    results = []
    for i, exp_conf in enumerate(experiments_to_run):
        print(f"\n--- Progress: Experiment {i+1} / {len(experiments_to_run)} ---")
        success = run_single_experiment(exp_conf, base_omega_conf) # type: ignore
        results.append({
            "model": exp_conf["name"],
            "status": "success" if success else "failed"
        })

    print("\nAll scheduled experiments have been attempted.")
    print("Please check WandB for detailed results and logs.")

    # Save results to CSV
    with open("validation_results.csv", "w", newline="") as csvfile:
        fieldnames = ["model", "status"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print("Validation results saved to validation_results.csv")

if __name__ == "__main__":
    # Ensure the script is run from the root of the repository for imports to work correctly
    # or adjust PYTHONPATH.
    # Example: python -m scripts.automate_training
    main()
