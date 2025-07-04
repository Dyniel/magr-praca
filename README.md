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

*   **Multiple Architectures**: Supports two main GAN architectures:
    *   `gan5_gcn`: Uses GCNs directly on superpixel features within the generator (based on `legacy/gan5.py`).
    *   `gan6_gat_cnn`: Employs a Graph Attention Network (GAT) based encoder to produce a graph embedding, which then conditions a standard CNN generator (based on `legacy/gan6.py`).
*   **Superpixel Processing**: Utilizes SLIC superpixels. For `gan5_gcn`, adjacency matrices are used. For `gan6_gat_cnn`, graph structures compatible with PyTorch Geometric are created (nodes are superpixels, edges from RAG).
*   **Configurable Components**: Model hyperparameters (channels, layers, dropout, GAT settings, etc.) are configurable for both architectures.
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
    A `requirements.txt` file is provided. Install the dependencies using:
    ```bash
    pip install -r requirements.txt
    ```
    If you encounter issues, especially with `torch-geometric`, refer to the manual installation notes below.

    **Key Dependencies (for manual installation or reference):**
    *   `torch>=1.10` (tested with 1.13.1)
    *   `torchvision`
    *   `numpy`
    *   `scikit-image`
    *   `Pillow`
    *   `omegaconf>=2.1`
    *   `tqdm`
    *   `wandb` (for logging, optional if `use_wandb=False`)
    *   `pytorch-fid` (for FID calculation)
    *   `torch-geometric` (required for `gan6_gat_cnn` architecture)

    *   **Manual Installation for PyTorch Geometric (`torch-geometric`):** Installation can be version-sensitive with PyTorch and CUDA. Follow the official instructions at [PyTorch Geometric's documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). You'll need to identify your PyTorch version and CUDA version (if using GPU). For example, if you have PyTorch 1.13.1 and CUDA 11.7, the command might be:
        ```bash
        pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
        pip install torch_geometric
        ```
        *Always check the PyG website for the most up-to-date commands for your specific environment.*

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
*   `image_size`: General image processing size.
*   **Model Architecture Selection**:
    *   `model.architecture`: Set to `"gan5_gcn"` or `"gan6_gat_cnn"` to choose the model type.
*   **`gan5_gcn` specific parameters (under `model.`):**
    *   `num_superpixels`, `slic_compactness` (used by `SuperpixelDataset` via top-level config forwarding by Trainer for now, or could be moved under `model.gan5_params`).
    *   `z_dim`, `g_channels`, `g_num_gcn_blocks`, `d_channels`, etc.
*   **`gan6_gat_cnn` specific parameters (under `model.`):**
    *   `model.gan6_num_superpixels`, `model.gan6_slic_compactness` (used by `ImageToGraphDataset`).
    *   `model.gat_dim`, `model.gat_heads`, `model.gat_layers`, `model.gan6_z_dim_graph_encoder_output`.
    *   `model.gan6_z_dim_noise`, `model.gan6_gen_init_size`, `model.gan6_gen_feat_start`.
    *   `model.gan6_d_feat_start`, `model.gan6_d_final_conv_size`.
*   Training hyperparameters: `batch_size`, `num_epochs`, learning rates (`g_lr`, `d_lr`), `r1_gamma`, etc. (Note: For `gan6`, `g_lr` is used for both G and E optimizers).
*   Logging: `use_wandb`, `log_freq_step`, `sample_freq_epoch`.
*   FID Calculation: `enable_fid_calculation`, `fid_num_images`, `fid_batch_size`, `fid_freq_epoch`.
*   **Example for `gan6`**: Refer to `configs/experiment_gan6_config.yaml`.

## Training

To start training, run the `scripts/train.py` script. You'll need to specify a configuration file.

**Example for a "Proper" Experiment (using `gan5_gcn` defaults from `experiment_config.yaml`):**

This command uses the settings in `configs/experiment_config.yaml` (which we've updated to use all images, 250 epochs, batch size 8, and enabled Weights & Biases).

```bash
python -m scripts.train --config_file configs/experiment_config.yaml
```

**Customizing and Overriding:**

*   `--config_file <path>`: Path to your experiment's YAML configuration file.
    *   For `gan5_gcn` based experiments, start with `configs/experiment_config.yaml`.
    *   For `gan6_gat_cnn` based experiments, start with `configs/experiment_gan6_config.yaml`.
*   `[additional_overrides]`: Optional command-line overrides for any configuration parameter, in `key=value` format. These will take precedence over values in the YAML file.
    For example, for a quick debug run with few epochs and images, and wandb disabled:
    ```bash
    python -m scripts.train --config_file configs/experiment_config.yaml num_epochs=5 debug_num_images=10 use_wandb=False
    ```

**Enabling/Disabling Weights & Biases (WandB):**

*   WandB logging is controlled by the `use_wandb` parameter in your configuration.
*   It is set to `True` by default in `configs/base_config.py` and in the example experiment YAML files.
*   To disable WandB for a specific run, add `use_wandb=False` to your command-line arguments.

**Resuming Training:**

To resume training from a saved checkpoint:
1.  Set the `resume_checkpoint_path` in your YAML configuration file to the path of your `.pth.tar` checkpoint.
    ```yaml
    # In your_experiment_config.yaml
    resume_checkpoint_path: "experiment_outputs/SuperpixelGAN_LungSCC_Experiment1/gan5_refactored_lr_1e-5_bs_8/checkpoints/checkpoint_epoch_0000.pth.tar" # Update with your actual project/run name and epoch
    ```
2.  Then run the training script as usual with that config file:
    ```bash
    python -m scripts.train --config_file configs/your_experiment_config.yaml
    ```
*   Ensure that the configuration (especially model architecture and key dimensions) used for resuming is compatible with the checkpoint. The configuration saved within the checkpoint is primarily for reference.

**Output:**

*   **Checkpoints**: Saved to `results/<project_name>/<run_name>/checkpoints/`.
*   **Generated Samples**: Saved to `results/<project_name>/<run_name>/samples/`.
*   **WandB Logs**: If `use_wandb` is true, logs will be sent to your Weights & Biases account under the specified project and run name.
*   **FID Scores**: Logged to console and WandB if enabled. Temporary images for FID are stored in `results/<project_name>/<run_name>/fid_*_images_temp/`.

## Superpixel Caching

*   The dataset classes (`SuperpixelDataset` for `gan5_gcn`, `ImageToGraphDataset` for `gan6_gat_cnn`) handle caching of precomputed data.
    *   For `gan5_gcn`: Segmentations and adjacency matrices are cached in `config.cache_dir/sp_<num_superpixels>_is_<image_size>/`.
    *   For `gan6_gat_cnn`: PyTorch Geometric `Data` objects (graphs) are cached in `config.cache_dir/pyg_graphs_sp<model.gan6_num_superpixels>_slic<model.gan6_slic_compactness>_is<image_size>/`.
*   If the cache directory is not found or seems incomplete for the current image paths and parameters, precomputation will run automatically. This can take time for large datasets.
*   The cache is specific to the dataset parameters. If you change these parameters (e.g., `num_superpixels`, `image_size`), a new cache will be generated.

## Further Development

*   **More Sophisticated Model Variants**:
    *   Explore different GCN/GAT architectures or attention mechanisms.
*   **Advanced Data Augmentation (ADA)**.
*   **Hyperparameter Optimization**: Use tools like WandB Sweeps or Optuna.
*   **Unit and Integration Tests**: Add tests for data loading, model components, and the training loop.

```
