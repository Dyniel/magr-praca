import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import relabel_sequential
import os
import shutil
from tqdm import tqdm
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float

# From legacy/gan5.py
# ==================== UTILS ====================
def spectral_norm(layer):
    """Applies spectral normalization to a layer."""
    return nn.utils.spectral_norm(layer)


def relabel_and_clip(seg, max_labels):
    """Relabels segmentation mask and clips labels to max_labels."""
    seg, _, _ = relabel_sequential(seg)
    seg = np.where(seg < max_labels, seg, max_labels - 1)
    return seg.astype(np.int32)


def create_adjacency_matrix(seg_array, num_superpixels):
    """
    Creates a normalized adjacency matrix from a segmentation array.

    Args:
        seg_array (np.ndarray): Segmentation mask (H, W).
        num_superpixels (int): Number of superpixels (S).

    Returns:
        np.ndarray: Normalized adjacency matrix (S, S).
    """
    H, W = seg_array.shape
    S = num_superpixels
    adj = np.zeros((S, S), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            current_superpixel_label = int(seg_array[y, x])
            if current_superpixel_label >= S:  # Should not happen if relabel_and_clip was used
                continue

            # Check neighbors (up, down, left, right)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_superpixel_label = int(seg_array[ny, nx])
                    if neighbor_superpixel_label < S and neighbor_superpixel_label != current_superpixel_label:
                        adj[current_superpixel_label, neighbor_superpixel_label] = 1.0
                        adj[neighbor_superpixel_label, current_superpixel_label] = 1.0 # Symmetric

    # Normalize adjacency matrix
    deg = adj.sum(axis=1)
    inv_sqrt_deg = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt_deg[mask] = deg[mask] ** -0.5
    D_inv_sqrt = np.diag(inv_sqrt_deg)

    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt
    return A_hat

def precompute_superpixels_for_dataset(image_paths, cache_dir, image_size, num_superpixels_config, compactness=10):
    """
    Precomputes superpixels, features, and adjacency matrices for all images in a list.
    Saves them to cache_dir. This function is intended to be called once before training.
    It uses the num_superpixels from the config.
    """
    if os.path.exists(cache_dir):
        print(f"Cache directory {cache_dir} exists. Removing old cache.")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Preprocessing images and caching to {cache_dir}...")

    for img_path in tqdm(image_paths, desc="Preprocessing Images"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_file_path = os.path.join(cache_dir, base_name + ".npz")

        # This check is mostly redundant if we clear the cache, but good for robustness
        if os.path.exists(cache_file_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((image_size, image_size), Image.BILINEAR)

            # Segmentation
            img_float = img_as_float(img_resized)
            seg_mask = slic(img_float, n_segments=num_superpixels_config, compactness=compactness, start_label=0)
            seg_mask = relabel_and_clip(seg_mask, num_superpixels_config) # Ensure labels are 0 to S-1

            # Adjacency Matrix
            adj_matrix = create_adjacency_matrix(seg_mask, num_superpixels_config)

            # Note: Mean color features will be computed on-the-fly by the Dataset's __getitem__
            # to avoid storing potentially large raw image data or feature data if not always needed
            # in the same format. Here we only cache segmentation and adjacency.
            np.savez(cache_file_path, segments=seg_mask, adj=adj_matrix)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    print("Preprocessing complete.")


# Placeholder for other utilities like:
# - Checkpoint saving/loading
# - Logging setup (e.g., for standard Python logger)
# - Image grid creation for visualization (if not using torchvision.utils directly)

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    """Saves model and optimizer state."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def load_checkpoint(checkpoint_path, model_g, model_d, optimizer_g=None, optimizer_d=None, device='cpu'):
    """
    Loads model and optimizer states from a checkpoint file for Generator and Discriminator.
    Returns the epoch and step to resume from.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, 0 # Default to starting from scratch

    print(f"Loading checkpoint from {checkpoint_path}...")
    # Load checkpoint to the specified device to avoid issues if saved on GPU and loading on CPU
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'G_state_dict' in checkpoint:
        model_g.load_state_dict(checkpoint['G_state_dict'])
    else:
        print("Warning: Generator state_dict not found in checkpoint.")
        return 0,0

    if 'D_state_dict' in checkpoint:
        model_d.load_state_dict(checkpoint['D_state_dict'])
    else:
        print("Warning: Discriminator state_dict not found in checkpoint.")
        return 0,0

    if optimizer_g and 'optG_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optG_state_dict'])
    elif optimizer_g:
        print("Warning: Generator optimizer state_dict not found in checkpoint.")

    if optimizer_d and 'optD_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optD_state_dict'])
    elif optimizer_d:
        print("Warning: Discriminator optimizer state_dict not found in checkpoint.")

    start_epoch = checkpoint.get('epoch', 0)
    current_step = checkpoint.get('step', 0) # Renamed from 'current_step' for clarity

    # If resuming, typically start from the next epoch
    # However, if saving mid-epoch, step is more accurate.
    # Trainer will handle epoch increment logic.
    print(f"Resuming from Epoch: {start_epoch}, Step: {current_step}")

    # Return epoch and step. Trainer should probably start from epoch+1 if step is 0,
    # or continue from current_epoch if step > 0 from a mid-epoch save.
    # For simplicity, let trainer handle this. We return what's in checkpoint.
    return start_epoch, current_step


def setup_wandb(config, model, project_name="MedicalR3GAN_Refactored", watch_model=True):
    """Initializes Weights & Biases if enabled in config."""
    if config.use_wandb:
        try:
            import wandb
            wandb.init(project=project_name, config=vars(config))
            if watch_model and model is not None:
                wandb.watch(model)
            print("Weights & Biases initialized.")
            return wandb
        except ImportError:
            print("wandb not installed. Skipping W&B initialization.")
            config.use_wandb = False # Disable if import fails
            return None
        except Exception as e:
            print(f"Could not initialize W&B: {e}. Skipping W&B initialization.")
            config.use_wandb = False # Disable on other errors
            return None
    return None

def log_to_wandb(wandb_run, metrics_dict, step=None):
    """Logs metrics to W&B if enabled and initialized."""
    if wandb_run:
        if step is not None:
            wandb_run.log(metrics_dict, step=step)
        else:
            wandb_run.log(metrics_dict)

# Ensure cfg is not directly used in utils. It should be passed or accessed via config object.
# The create_adjacency_matrix was dependent on cfg.num_superpixels.
# I've changed its signature to accept num_superpixels directly.
# The precompute_superpixels also depended on cfg. I've updated its signature.
# It now takes num_superpixels_config.
# The original precompute_superpixels function in gan5.py also had image_size and cache_dir from cfg.
# These are now passed as arguments.
# It also directly used cfg.num_superpixels, which is now num_superpixels_config.

# Consider adding a function to prepare image paths from a directory.
def get_image_paths(dataset_dir, extensions=('.png', '.jpg', '.jpeg')):
    """Gets all image paths from a directory with given extensions."""
    paths = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(extensions):
                paths.append(os.path.join(root, file))
    return sorted(paths)

# Add a simple function for image normalization to be used by dataset
# if not using torchvision transforms directly for everything.
def normalize_image(image_tensor):
    """Normalizes a tensor image to [-1, 1]. Assumes input is [0, 1]."""
    return image_tensor * 2.0 - 1.0

def denormalize_image(image_tensor):
    """Denormalizes a tensor image from [-1, 1] to [0, 1]."""
    return (image_tensor + 1.0) / 2.0

class ImageToTensor:
    """Converts a PIL Image to a PyTorch tensor (C, H, W) in range [0, 1]."""
    def __call__(self, pil_image):
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        if img_np.ndim == 2: # Grayscale
            img_np = np.expand_dims(img_np, axis=-1)
        return torch.from_numpy(img_np.transpose(2, 0, 1))

class ResizePIL:
    """Resizes a PIL image."""
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pil_image):
        return pil_image.resize(self.size, self.interpolation)

print("src/utils.py created and populated.")
