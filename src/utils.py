import torch
from torch.nn.utils import spectral_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import spectral_norm
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import wandb
from torchvision.utils import make_grid
import numpy as np
from skimage.segmentation import relabel_sequential
import os
import shutil
from tqdm import tqdm
from PIL import Image
from skimage.segmentation import slic
from skimage.util import img_as_float

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
            if current_superpixel_label >= S:
                continue

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_superpixel_label = int(seg_array[ny, nx])
                    if neighbor_superpixel_label < S and neighbor_superpixel_label != current_superpixel_label:
                        adj[current_superpixel_label, neighbor_superpixel_label] = 1.0
                        adj[neighbor_superpixel_label, current_superpixel_label] = 1.0

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

def load_checkpoint(checkpoint_path, model_g, model_d, model_e=None,
                    optimizer_g=None, optimizer_d=None, optimizer_e=None,
                    device='cpu'):
    """
    Loads model and optimizer states from a checkpoint file.
    Handles optional Encoder model (model_e) and its optimizer (optimizer_e) for gan6.
    Returns the epoch and step to resume from.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return 0, 0 # Default to starting from scratch

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load Generator
    if 'G_state_dict' in checkpoint and model_g is not None:
        model_g.load_state_dict(checkpoint['G_state_dict'])
    elif model_g is not None:
        print("Warning: Generator state_dict (G_state_dict) not found in checkpoint.")
        # return 0,0 # Or allow partial load? For now, require G and D.

    # Load Discriminator
    if 'D_state_dict' in checkpoint and model_d is not None:
        model_d.load_state_dict(checkpoint['D_state_dict'])
    elif model_d is not None:
        print("Warning: Discriminator state_dict (D_state_dict) not found in checkpoint.")
        # return 0,0

    # Load Encoder (optional, for gan6)
    if model_e is not None: # Only try to load if model_e is provided
        if 'E_state_dict' in checkpoint:
            model_e.load_state_dict(checkpoint['E_state_dict'])
        else:
            print("Warning: Encoder state_dict (E_state_dict) not found in checkpoint, but an Encoder model was provided.")

    # Load Optimizers
    if optimizer_g and 'optG_state_dict' in checkpoint:
        optimizer_g.load_state_dict(checkpoint['optG_state_dict'])
    elif optimizer_g:
        print("Warning: Generator optimizer state_dict (optG_state_dict) not found in checkpoint.")

    if optimizer_d and 'optD_state_dict' in checkpoint:
        optimizer_d.load_state_dict(checkpoint['optD_state_dict'])
    elif optimizer_d:
        print("Warning: Discriminator optimizer state_dict (optD_state_dict) not found in checkpoint.")

    if optimizer_e and model_e is not None: # Only try to load if optimizer_e and model_e are provided
        if 'optE_state_dict' in checkpoint:
            optimizer_e.load_state_dict(checkpoint['optE_state_dict'])
        else:
            print("Warning: Encoder optimizer state_dict (optE_state_dict) not found in checkpoint, but an Encoder optimizer was provided.")

    start_epoch = checkpoint.get('epoch', 0)
    current_step = checkpoint.get('step', 0)

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
    def __init__(self, size, interpolation=Image.BILINEAR): # Ensure PIL.Image is imported as Image
        self.size = size
        self.interpolation = interpolation

    def __call__(self, pil_image):
        if not isinstance(pil_image, Image.Image):
            # Try to get the path if available (might not be directly in transform context)
            # This is a best-effort log. The path is more reliably logged in __getitem__.
            print(f"ERROR: ResizePIL received an invalid object instead of a PIL image. Type: {type(pil_image)}. Object: {str(pil_image)[:100]}")
            # Option: raise an error here, or return None, or return the object as is.
            # Returning the object as is will likely cause a downstream error, which is fine for now.
            # If we returned None, the next transform might fail or handle it.
            # Raising an error here would be more explicit.
            raise TypeError(f"ResizePIL expected a PIL.Image.Image object, but got {type(pil_image)}")
        return pil_image.resize(self.size, self.interpolation)

# --- PyTorch Geometric Related Utilities (for gan6 architecture) ---
try:
    from torch_geometric.data import Data
    import skimage.graph as skgraph # For RAG
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    # print("Warning: PyTorch Geometric or scikit-image not fully available. Graph conversion utilities might fail.")
    Data = None # Define Data as None or a placeholder if pyg not installed
    skgraph = None


def convert_image_to_pyg_graph(image_numpy_array_01, num_superpixels, slic_compactness=10, feature_type='mean_color'):
    """
    Converts a single image to a PyTorch Geometric Data object.
    """
    if not PYG_AVAILABLE or Data is None or skgraph is None:
        raise ImportError("PyTorch Geometric or scikit-image.graph is required for graph conversion.")

    spx_labels = slic(image_numpy_array_01, n_segments=num_superpixels, compactness=slic_compactness, start_label=0)
    num_actual_nodes = np.max(spx_labels) + 1

    if feature_type == 'mean_color':
        node_features = np.zeros((num_actual_nodes, image_numpy_array_01.shape[2]), dtype=np.float32)
        for c in range(image_numpy_array_01.shape[2]):
            channel_sum_per_superpixel = np.bincount(spx_labels.flatten(), weights=image_numpy_array_01[..., c].flatten(), minlength=num_actual_nodes)
            pixel_counts_per_superpixel = np.bincount(spx_labels.flatten(), minlength=num_actual_nodes)
            valid_mask = pixel_counts_per_superpixel > 0
            node_features[valid_mask, c] = channel_sum_per_superpixel[valid_mask] / pixel_counts_per_superpixel[valid_mask]
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    rag = skgraph.rag_mean_color(image_numpy_array_01, spx_labels)
    edge_list = np.array(list(rag.edges), dtype=np.int64).T if len(rag.edges) > 0 else np.empty((2, 0), dtype=np.int64)
    edge_index = torch.from_numpy(edge_list)
    x = torch.from_numpy(node_features)
    adj = create_adjacency_matrix(spx_labels, num_actual_nodes)
    return Data(x=x, edge_index=edge_index, adj=torch.from_numpy(adj))

def generate_spatial_superpixel_map(segments_map_batch, num_channels_out, image_size, num_superpixels_total, real_images_batch_01=None):
    """
    Generates a spatial map representation of superpixels.
    Args:
        segments_map_batch (torch.Tensor): Batch of segmentation masks [B, H, W] with superpixel IDs.
        num_channels_out (int): Desired number of output channels for the map.
                                (e.g., 1 for normalized IDs, 3 for mean color, S for one-hot (not recommended for large S)).
        image_size (int): Target H and W for the output map.
        num_superpixels_total (int): The maximum number of superpixels segments_map_batch IDs go up to. Used for normalization or one-hot.
        real_images_batch_01 (torch.Tensor, optional): Batch of real images [B, 3, H, W] in [0,1] range,
                                                     used if num_channels_out=3 (mean color).
    Returns:
        torch.Tensor: Spatial superpixel map [B, num_channels_out, H, W].
    """
    B, H, W = segments_map_batch.shape
    device = segments_map_batch.device
    output_map = torch.zeros(B, num_channels_out, H, W, device=device)

    if H != image_size or W != image_size:
        # This case should ideally be avoided by ensuring segments_map_batch matches target image_size.
        # If not, we might need to resize segments_map_batch (nearest neighbor) and real_images_batch_01 (bilinear).
        # For simplicity, assuming H, W already match image_size.
        pass


    if num_channels_out == 1: # Normalized segment IDs
        # Normalize segment IDs to [0, 1] or [-1, 1] - let's use [0,1] for now.
        # This is a very basic representation.
        normalized_segments = segments_map_batch.float() / (num_superpixels_total - 1)
        output_map = normalized_segments.unsqueeze(1) # [B, 1, H, W]

    elif num_channels_out == 3 and real_images_batch_01 is not None: # Mean color map
        if real_images_batch_01.shape[0] != B or real_images_batch_01.shape[2] != H or real_images_batch_01.shape[3] != W:
            raise ValueError("Real images batch dimensions mismatch segments map batch for mean color map.")

        for b_idx in range(B):
            img_single = real_images_batch_01[b_idx] # [3, H, W]
            seg_single = segments_map_batch[b_idx]   # [H, W]

            unique_sp_ids = torch.unique(seg_single)
            mean_colors_sp = torch.zeros(num_superpixels_total, 3, device=device) # Max S possible IDs

            for sp_id_val in unique_sp_ids:
                sp_id = sp_id_val.item()
                if sp_id >= num_superpixels_total: continue # Should not happen with good segmentation
                mask = (seg_single == sp_id) # [H, W]
                if mask.sum() > 0:
                    # Calculate mean color for this superpixel
                    # img_single is [3, H, W], mask is [H,W]
                    # We want to select pixels from img_single where mask is true, then mean over selected pixels for each channel
                    for ch in range(3):
                        mean_colors_sp[sp_id, ch] = img_single[ch][mask].mean()

            # Scatter mean colors back to the image grid
            # output_map[b_idx] is [3, H, W]
            # seg_single_long = seg_single.long() # Ensure it's long for indexing
            # This scatter is tricky. A simpler way for small S: iterate unique_sp_ids
            for sp_id_val in unique_sp_ids:
                sp_id = sp_id_val.item()
                if sp_id >= num_superpixels_total: continue
                mask = (seg_single == sp_id)
                for ch in range(3):
                    output_map[b_idx, ch, mask] = mean_colors_sp[sp_id, ch]

    elif num_channels_out == num_superpixels_total: # One-hot encoding (careful with memory for large S)
        # This is memory intensive if S is large (e.g. 150 channels for 256x256 image)
        # F.one_hot expects class indices, so segments_map_batch needs to be long
        one_hot = F.one_hot(segments_map_batch.long(), num_classes=num_superpixels_total) # [B, H, W, S]
        output_map = one_hot.permute(0, 3, 1, 2).float() # [B, S, H, W]

    else:
        print(f"Warning: generate_spatial_superpixel_map unsupported num_channels_out={num_channels_out}. Returning zeros.")

    return output_map

def calculate_mean_superpixel_features(images_batch_01, segments_map_batch, num_superpixels_total, feature_dim=3):
    """
    Calculates mean features (e.g., RGB color) for each superpixel in a batch.
    Args:
        images_batch_01 (torch.Tensor): Batch of images [B, C, H, W], normalized to [0,1]. C should match feature_dim.
        segments_map_batch (torch.Tensor): Batch of segmentation masks [B, H, W].
        num_superpixels_total (int): Max number of superpixels S.
        feature_dim (int): Dimensionality of features to calculate (e.g., 3 for RGB).
    Returns:
        torch.Tensor: Mean features per superpixel [B, S, feature_dim].
    """
    B, C, H, W = images_batch_01.shape
    device = images_batch_01.device

    if C != feature_dim:
        raise ValueError(f"Image channels {C} must match feature_dim {feature_dim} for mean feature calculation.")

    all_mean_features = torch.zeros(B, num_superpixels_total, feature_dim, device=device)

    for b_idx in range(B):
        img_single = images_batch_01[b_idx]  # [C, H, W]
        seg_single = segments_map_batch[b_idx] # [H, W]

        # Ensure seg_single is on the same device and long type for bincount/unique
        seg_single_flat = seg_single.flatten().long()

        for c_idx in range(feature_dim):
            img_channel_flat = img_single[c_idx].flatten()
            # Sum features per superpixel using bincount weights
            sum_feats_per_sp = torch.bincount(seg_single_flat, weights=img_channel_flat, minlength=num_superpixels_total)
            # Count pixels per superpixel
            pixel_counts_per_sp = torch.bincount(seg_single_flat, minlength=num_superpixels_total)

            # Avoid division by zero for superpixels not present or empty
            valid_mask = pixel_counts_per_sp > 0
            mean_feats_this_channel = torch.zeros_like(sum_feats_per_sp, dtype=img_single.dtype)
            mean_feats_this_channel[valid_mask] = sum_feats_per_sp[valid_mask] / pixel_counts_per_sp[valid_mask]

            all_mean_features[b_idx, :, c_idx] = mean_feats_this_channel

    return all_mean_features


def toggle_grad(model, requires_grad):
    """
    Sets requires_grad for all parameters of a model.
    """
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def log_image_grid_to_wandb(images, wandb_run, caption, step):
    """
    Logs a grid of images to W&B.
    """
    if wandb_run:
        grid = make_grid(images)
        wandb_run.log({caption: [wandb.Image(grid, caption=caption)]}, step=step)


def interpolate_spatial_map(spatial_map, size):
    """
    Interpolates a spatial map to a given size.
    """
    if spatial_map is None:
        return None
    if spatial_map.shape[-2:] != (size, size):
        return F.interpolate(spatial_map, size=(size, size), mode='nearest')
    return spatial_map


print("src/utils.py created and populated.")
