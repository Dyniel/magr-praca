# Superpixel-Conditioned GAN (Refactored)

This project implements a Generative Adversarial Network (GAN) that is conditioned on superpixel information. The core idea is to leverage the structural information from superpixels and their graph relationships to improve the quality of generated images. This version is a refactor of an earlier codebase, focusing on the architecture similar to `legacy/gan5.py`.

## Project Structure

```
.
├── configs/
│   ├── base_config.py            # Dataclass with all default configuration parameters
│   └── experiment_config.yaml    # Example YAML to override base_config for specific experiments
├── data/
│   └── (Your dataset should be placed here or path specified in config)
├── legacy/
│   ├── gan.py                    # Original/legacy GAN implementations
│   └── ...
├── results/
│   └── (Output directory base, will contain subfolders for project/run)
├── scripts/
│   └── train.py                  # Main script to start training
├── src/
│   ├── data_loader.py            # SuperpixelDataset and DataLoader setup
│   ├── models.py                 # Generator, Discriminator, and NN building blocks
│   ├── trainer.py                # Trainer class orchestrating the training loop
│   └── utils.py                  # Utility functions (spectral norm, checkpointing, etc.)
├── README.md                     # This file
└── requirements.txt              # (Recommended to be added) Python dependencies
```

## Features

*   **Superpixel Conditioning**: Utilizes SLIC superpixels and their adjacency information.
*   **Graph Convolutional Networks (GCNs)**: The Generator uses GCN blocks to process superpixel features.
*   **Configurable Architecture**: Model hyperparameters (channels, layers, dropout, etc.) are configurable.
*   **Flexible Configuration System**: Uses OmegaConf to manage configurations via a base dataclass and YAML override files.
*   **Weights & Biases Logging**: Integrated for experiment tracking (can be disabled).
*   **Checkpointing**: Saves model checkpoints during training.
*   **Resume Training**: Allows resuming training from a saved checkpoint.
*   **Sample Generation**: Generates and logs image samples during training.
*   **FID Calculation**: Supports calculation of Fréchet Inception Distance for quantitative evaluation.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    It's recommended to create a `requirements.txt` file. Key dependencies include:
    *   `torch`
    *   `torchvision`
    *   `numpy`
    *   `scikit-image`
    *   `Pillow`
    *   `omegaconf`
    *   `tqdm`
    *   `wandb` (optional, for logging)
    *   `pytorch-fid` (for FID calculation)
    ```bash
    pip install torch torchvision numpy scikit-image Pillow omegaconf tqdm wandb pytorch-fid
    # Add other specific versions as needed
    ```

4.  **Dataset:**
    *   Download or prepare your image dataset.
    *   Update the `dataset_path` in `configs/base_config.py` or your experiment YAML file to point to your dataset directory. The default is `/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc`.

## Configuration

The training process is controlled by configuration files:

*   **`configs/base_config.py`**: Contains the `BaseConfig` dataclass with all default parameters. Review this file to understand all available options.
*   **`configs/experiment_config.yaml`**: This is an *example* YAML file. For a new experiment:
    1.  Copy `experiment_config.yaml` to a new file (e.g., `configs/my_experiment.yaml`).
    2.  Modify the parameters in `my_experiment.yaml` as needed. Parameters not specified in the YAML will take their default values from `BaseConfig`.

Key configuration aspects:
*   `project_name`, `run_name`: Define naming for output directories and WandB logs.
*   `dataset_path`, `cache_dir`: Paths for data and precomputed superpixels.
*   `image_size`, `num_superpixels`, `slic_compactness`: Data preprocessing parameters.
*   Model hyperparameters: `z_dim`, `g_channels`, `d_channels`, `g_num_gcn_blocks`, etc.
*   Training hyperparameters: `batch_size`, `num_epochs`, learning rates (`g_lr`, `d_lr`), `r1_gamma`, etc.
*   Logging: `use_wandb`, `log_freq_step`, `sample_freq_epoch`.
*   FID Calculation: `enable_fid_calculation`, `fid_num_images`, `fid_batch_size`, `fid_freq_epoch`.

## Training

To start training, run the `scripts/train.py` script:

```bash
python scripts/train.py --config_file configs/your_experiment_config.yaml [additional_overrides]
```

**Arguments:**

*   `--config_file <path>`: Path to your experiment's YAML configuration file.
    (Defaults to `configs/experiment_config.yaml`).
*   `[additional_overrides]`: Optional command-line overrides for configuration parameters, in `key=value` format.
    For example: `python scripts/train.py batch_size=8 num_epochs=10`

**To resume training from a checkpoint:**

```bash
python scripts/train.py --config_file configs/your_experiment_config.yaml --resume_from results/<project_name>/<run_name>/checkpoints/checkpoint_epoch_xxxx.pth.tar
```
*   The `--resume_from` argument specifies the path to the checkpoint file.
*   You can also set `resume_checkpoint_path` in your YAML configuration file. The command-line argument takes precedence.
*   Ensure that the configuration (especially model architecture) used for resuming is compatible with the checkpoint. The saved configuration within the checkpoint is for reference but not automatically reapplied.

**Output:**

*   **Checkpoints**: Saved to `results/<project_name>/<run_name>/checkpoints/`.
*   **Generated Samples**: Saved to `results/<project_name>/<run_name>/samples/`.
*   **WandB Logs**: If `use_wandb` is true, logs will be sent to your Weights & Biases account under the specified project and run name.
*   **FID Scores**: Logged to console and WandB if enabled. Temporary images for FID are stored in `results/<project_name>/<run_name>/fid_*_images_temp/`.

## Superpixel Caching

*   The `SuperpixelDataset` precomputes superpixel segmentations and adjacency matrices.
*   These are cached in the directory specified by `config.cache_dir` (within which a subdirectory like `sp_<num_superpixels>_is_<image_size>` is created).
*   If the cache directory is not found or seems incomplete for the current image paths and parameters, precomputation will run automatically. This can take time for large datasets.
*   The cache is specific to `num_superpixels` and `image_size`. If you change these parameters, a new cache will be generated.

## Further Development

*   **More Sophisticated Model Variants**:
    *   Integrate ideas from `legacy/gan6.py` (Graph Encoder producing `z_graph`). This is a planned next step.
    *   Explore different GCN architectures or attention mechanisms.
*   **Advanced Data Augmentation (ADA)**.
*   **Hyperparameter Optimization**: Use tools like WandB Sweeps or Optuna.
*   **Unit and Integration Tests**: Add tests for data loading, model components, and the training loop.

```
