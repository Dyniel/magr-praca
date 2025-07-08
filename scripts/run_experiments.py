import subprocess
import tempfile
import os
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
    base_architectures = ["gan5_gcn", "gan6_gat_cnn", "dcgan", "stylegan2", "projectedgan", "stylegan3"]

    for arch in base_architectures:
        # --- Standard Run (Superpixel Conditioning Disabled, or native if model requires it) ---
        std_overrides = {
            "model.architecture": arch,
            "model.use_superpixel_conditioning": False, # Default to off for non-superpixel native models
            "run_name": f"{arch}_standard_no_sp_cond",
             # Reduce epochs for faster testing of the script itself. User should increase for real runs.
            "num_epochs": 2, # TODO: Remove or increase for actual experiments
            "debug_num_images": 16, # TODO: Remove or increase for actual experiments
            "batch_size": 2, # TODO: Remove or increase
            "enable_fid_calculation": False, # Disable FID for faster script testing
        }

        if arch == "gan5_gcn":
            # gan5_gcn inherently uses superpixels, so 'use_superpixel_conditioning' doesn't gate its core mechanism.
            # Its "standard" run is just its default behavior.
            std_overrides["run_name"] = f"{arch}_standard"
            std_overrides["model.gan5_gcn_disable_gcn_blocks"] = False
            # No need to set model.use_superpixel_conditioning for gan5/gan6 as their dataloaders
            # are intrinsically superpixel-based. The flag is more for add-on conditioning.
        elif arch == "gan6_gat_cnn":
            std_overrides["run_name"] = f"{arch}_standard"
            std_overrides["model.gan6_gat_cnn_use_null_graph_embedding"] = False

        experiments.append({
            "name": f"{arch}_Standard",
            "model_architecture": arch,
            "config_overrides": std_overrides.copy()
        })

        # --- Ablation Runs / Superpixel Conditioned Runs ---
        ablation_overrides = std_overrides.copy() # Start from standard settings

        if arch == "gan5_gcn":
            ablation_overrides["model.gan5_gcn_disable_gcn_blocks"] = True
            ablation_overrides["run_name"] = f"{arch}_ablation_no_gcn"
            experiments.append({
                "name": f"{arch}_Ablation_NoGCN",
                "model_architecture": arch,
                "config_overrides": ablation_overrides
            })
        elif arch == "gan6_gat_cnn":
            ablation_overrides["model.gan6_gat_cnn_use_null_graph_embedding"] = True
            ablation_overrides["run_name"] = f"{arch}_ablation_null_graph_embed"
            experiments.append({
                "name": f"{arch}_Ablation_NullGraphEmbedding",
                "model_architecture": arch,
                "config_overrides": ablation_overrides
            })
        else: # For DCGAN, StyleGAN2, StyleGAN3, ProjectedGAN - "ablation" is enabling superpixel conditioning
            # These are models where superpixel conditioning is an add-on.
            # The "standard" run defined above has model.use_superpixel_conditioning=False.
            # This "conditioned" run will enable it.
            cond_overrides = std_overrides.copy()
            cond_overrides["model.use_superpixel_conditioning"] = True
            cond_overrides["run_name"] = f"{arch}_conditioned_sp_spatial_latent" # Example name

            # Enable specific conditioning types for these models (can be fine-tuned)
            # C1: Spatial G, C4: Spatial D, C2: Latent G
            # Ensure model config has these flags: e.g. model.dcgan_g_spatial_cond
            cond_overrides[f"model.{arch}_g_spatial_cond"] = True
            cond_overrides[f"model.{arch}_d_spatial_cond"] = True
            cond_overrides[f"model.{arch}_g_latent_cond"] = True # Requires sp_latent_encoder
            cond_overrides["model.superpixel_latent_encoder_enabled"] = True # Enable the encoder itself

            # Special handling for ProjectedGAN D conditioning (doesn't have g_latent_cond at G level like others)
            if arch == "projectedgan":
                 cond_overrides[f"model.stylegan2_g_spatial_cond"] = True # PG G is StyleGAN2
                 cond_overrides[f"model.stylegan2_g_latent_cond"] = True
                 cond_overrides[f"model.projectedgan_d_spatial_cond"] = True
                 # Remove generic arch flags if they don't exist for projectedgan directly
                 cond_overrides.pop(f"model.{arch}_g_spatial_cond", None)
                 cond_overrides.pop(f"model.{arch}_g_latent_cond", None)


            experiments.append({
                "name": f"{arch}_Conditioned_Superpixel",
                "model_architecture": arch,
                "config_overrides": cond_overrides
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
    conf.use_wandb = True # Force WandB for these experiments
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
        else:
            print(f"Error: Experiment {exp_config['name']} (Run: {conf.run_name}) failed with return code {process.returncode}.")

    except Exception as e:
        print(f"An error occurred while setting up or running experiment {exp_config['name']}: {e}")
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

    for i, exp_conf in enumerate(experiments_to_run):
        print(f"\n--- Progress: Experiment {i+1} / {len(experiments_to_run)} ---")
        run_single_experiment(exp_conf, base_omega_conf) # type: ignore

    print("\nAll scheduled experiments have been attempted.")
    print("Please check WandB for detailed results and logs.")

if __name__ == "__main__":
    # Ensure the script is run from the root of the repository for imports to work correctly
    # or adjust PYTHONPATH.
    # Example: python -m scripts.run_experiments
    main()
