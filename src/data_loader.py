import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms  # Will be used for basic transforms if not passed custom ones
from tqdm import tqdm

from src.utils import (
    precompute_superpixels_for_dataset, get_image_paths,
    ImageToTensor, ResizePIL, normalize_image
    # Removed: convert_image_to_pyg_graph, PYG_AVAILABLE
)

# Removed PyTorch Geometric imports as ImageToGraphDataset is being removed
# if PYG_AVAILABLE:
#     from torch_geometric.data import Batch as PyGBatch
#     from torch_geometric.data import Data as PyGData
# else:
#     PyGBatch = None
#     PyGData = None


class SuperpixelDataset(Dataset):  # For models needing precomputed superpixels, segments, adjacencies
    def __init__(self, image_paths, config, data_split_name="train", transform=None, target_transform=None):
        """
        Args:
            image_paths (list): List of paths to images.
            config (object): Configuration object with attributes like image_size,
                             num_superpixels, cache_dir.
            data_split_name (str): Name of the data split (e.g., "train", "val", "test") for cache naming.
            transform (callable, optional): Optional transform to be applied on a sample image.
            target_transform (callable, optional): Optional transform to be applied on segmentation/adj.
        """
        self.image_paths = image_paths
        self.config = config
        self.cache_dir = os.path.join(config.cache_dir,
                                      f"{data_split_name}_sp_{config.num_superpixels}_is_{config.image_size}")

        if transform is None:
            self.transform = transforms.Compose([
                ResizePIL((config.image_size, config.image_size)),
                ImageToTensor(),
                transforms.Lambda(normalize_image)
            ])
        else:
            self.transform = transform

        self.target_transform = target_transform

        expected_compactness = getattr(config, 'slic_compactness', 10.0)  # Default if not in config

        if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
            print(
                f"Cache directory {self.cache_dir} for split '{data_split_name}' not found or empty. Precomputing superpixels...")
            if not self.image_paths and hasattr(config, 'dataset_path') and data_split_name == "train":
                self.image_paths = get_image_paths(config.dataset_path)
            elif not self.image_paths and hasattr(config, 'dataset_path_val') and data_split_name == "val":
                self.image_paths = get_image_paths(config.dataset_path_val)
            elif not self.image_paths and hasattr(config, 'dataset_path_test') and data_split_name == "test":
                self.image_paths = get_image_paths(config.dataset_path_test)

            if not self.image_paths:
                raise ValueError(f"No image paths provided or found for split '{data_split_name}' for precomputation.")

            precompute_superpixels_for_dataset(
                image_paths=self.image_paths,
                cache_dir=self.cache_dir,
                image_size=config.image_size,
                num_superpixels_config=config.num_superpixels,
                compactness=expected_compactness
            )
        else:
            print(f"Using existing cache directory for split '{data_split_name}': {self.cache_dir}")
            missing_files = False
            for img_path in self.image_paths:  # Ensure image_paths are populated before this loop
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cache_file_path = os.path.join(self.cache_dir, base_name + ".npz")
                if not os.path.exists(cache_file_path):
                    print(f"Cache file missing for {img_path} at {cache_file_path}")
                    missing_files = True
                    break
            if missing_files:
                print(f"Some cache files are missing for split '{data_split_name}'. Re-running precomputation.")
                # Repopulate image_paths if they were empty, specific to split
                current_dataset_path = None
                if data_split_name == "train":
                    current_dataset_path = getattr(config, 'dataset_path', None)
                elif data_split_name == "val":
                    current_dataset_path = getattr(config, 'dataset_path_val', None)
                elif data_split_name == "test":
                    current_dataset_path = getattr(config, 'dataset_path_test', None)

                if not self.image_paths and current_dataset_path:
                    self.image_paths = get_image_paths(current_dataset_path)

                if not self.image_paths:
                    raise ValueError(f"No image paths found for split '{data_split_name}' to re-run precomputation.")

                precompute_superpixels_for_dataset(
                    image_paths=self.image_paths,
                    cache_dir=self.cache_dir,
                    image_size=config.image_size,
                    num_superpixels_config=config.num_superpixels,
                    compactness=expected_compactness
                )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_file_path = os.path.join(self.cache_dir, base_name + ".npz")

        image_tensor, segments, adj_matrix = None, None, None  # Initialize

        try:
            # Load cached segments and adjacency matrix
            try:
                data = np.load(cache_file_path)
                segments = torch.from_numpy(data["segments"]).long()
                adj_matrix = torch.from_numpy(data["adj"]).float()
            except FileNotFoundError:
                print(f"ERROR: SuperpixelDataset - Cache file not found for {img_path} at {cache_file_path}.")
                return None  # Indicate failure for this sample
            except Exception as e:
                print(
                    f"ERROR: SuperpixelDataset - Error loading cached data for {img_path} from {cache_file_path}: {e}")
                return None  # Indicate failure

            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"ERROR: SuperpixelDataset - Image file not found: {img_path}")
                return None
            except Exception as e:
                print(f"ERROR: SuperpixelDataset - Error loading image {img_path}: {e}")
                return None

            # Transform image
            if self.transform:
                try:
                    image_tensor = self.transform(image)
                except Exception as e:
                    print(f"ERROR: SuperpixelDataset - Error applying transform to image {img_path}: {e}")
                    # This could be where the ResizePIL error (now TypeError) is caught
                    return None
            else:  # Should always have a transform, but as a fallback
                image_tensor = image  # Or handle as error if transform is mandatory

            # Target transform (usually not used for these)
            if self.target_transform:
                try:
                    segments = self.target_transform(segments)
                    adj_matrix = self.target_transform(adj_matrix)
                except Exception as e:
                    print(f"ERROR: SuperpixelDataset - Error applying target_transform for {img_path}: {e}")
                    return None

            # Check if any essential component is still None
            if image_tensor is None or segments is None or adj_matrix is None:
                print(
                    f"ERROR: SuperpixelDataset - One or more components are None for {img_path} after processing, returning None.")
                return None

            return {
                "image": image_tensor,
                "segments": segments,
                "adj": adj_matrix,
                "path": img_path
            }

        except Exception as e:
            # Catch-all for any unexpected errors within the top-level try for this item
            print(f"CRITICAL ERROR in SuperpixelDataset.__getitem__ for {img_path}: {e}. Returning None.")
            return None

# Removed ImageToGraphDataset and collate_graphs as gan6_gat_cnn is removed

def get_dataloader(config, data_split="train", shuffle=True, drop_last=True):
    print("DEBUG: src.data_loader module loaded and get_dataloader CALLED!")
    dataset_path = None
    if data_split == "train":
        dataset_path = config.dataset_path
    elif data_split == "val":
        dataset_path = getattr(config, 'dataset_path_val', None)
    elif data_split == "test":
        dataset_path = getattr(config, 'dataset_path_test', None)
    else:
        raise ValueError(f"Invalid data_split: {data_split}. Must be 'train', 'val', or 'test'.")

    if dataset_path is None:
        print(f"Dataset path for split '{data_split}' is not configured. DataLoader will not be created.")
        return None

    image_paths = get_image_paths(dataset_path)
    if not image_paths:
        print(f"No images found in {dataset_path} for split '{data_split}'. DataLoader will not be created.")
        return None

        # Apply debug_num_images only for training split for faster debugging cycles on other splits
    if data_split == "train" and config.debug_num_images > 0 and config.debug_num_images < len(image_paths):
        print(f"Using a subset of {config.debug_num_images} images for training debugging.")
        image_paths = image_paths[:config.debug_num_images]
    elif data_split != "train" and getattr(config, f"debug_num_images_{data_split}", 0) > 0:
        # Allow specific debug_num_images for val/test if defined in config
        num_debug = getattr(config, f"debug_num_images_{data_split}")
        if num_debug < len(image_paths):
            print(f"Using a subset of {num_debug} images for {data_split} split debugging.")
            image_paths = image_paths[:num_debug]

    dataset_type = getattr(config.model, "architecture", "gan5_gcn")
    use_sp_conditioning = getattr(config.model, "use_superpixel_conditioning", False) # Default to False if not present
    dataset = None
    collate_fn_to_use = None # Default collate_fn will be used if this remains None

    # Architectures that can run in a "standard" (non-superpixel-conditioned) mode
    standard_gan_architectures = ["dcgan", "stylegan2", "stylegan3", "projected_gan", "histogan"]

    # Removed 'gan5_gcn' and 'gan6_gat_cnn' specific paths.
    # Removed 'cyclegan' specific path.

    if dataset_type in standard_gan_architectures:
      
        if use_sp_conditioning:
            # These GANs, when conditioned, will use C1, C2, C4 which rely on superpixel segments
            # and mean features derived from them. So, SuperpixelDataset is appropriate.
            print(f"Using SuperpixelDataset for {data_split} ({dataset_type} architecture with superpixel conditioning).")
            dataset = SuperpixelDataset(image_paths=image_paths, config=config, data_split_name=data_split)
        else:
            # Standard GAN mode, no superpixel conditioning. Use the simple ImageDataset.
            print(f"Using ImageDataset for {data_split} ({dataset_type} architecture without superpixel conditioning).")
            dataset = ImageDataset(image_paths=image_paths, config=config, data_split_name=data_split)
    # elif dataset_type != "gan5_gcn" and dataset_type not in standard_gan_architectures and dataset_type != "gan6_gat_cnn": # Defensive check for truly unknown
    # Fallback or error for unknown architectures
    # Now, if it's not in standard_gan_architectures, it's an error.
    else:
        raise ValueError(f"Unsupported or unknown model.architecture: {dataset_type} in get_dataloader. Supported: {standard_gan_architectures}")


    # Determine batch size: use eval_batch_size for val/test if defined, else main batch_size
    current_batch_size = config.batch_size
    if data_split in ["val", "test"] and hasattr(config, "eval_batch_size") and config.eval_batch_size is not None:
        current_batch_size = config.eval_batch_size
        print(f"Using eval_batch_size: {current_batch_size} for {data_split} split.")

    # For val/test, drop_last is typically False to evaluate all samples
    effective_drop_last = drop_last if data_split == "train" else False

    dataloader = DataLoader(
        dataset,
        batch_size=current_batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=effective_drop_last,
        collate_fn=collate_fn_to_use
    )
    return dataloader


class ImageDataset(Dataset):
    """
    A simple dataset to load images from a list of paths.
    Applies basic transformations like resize and normalization.
    Returns a dictionary containing the image tensor and its path.
    """
    def __init__(self, image_paths, config, data_split_name="train", transform=None):
        self.image_paths = image_paths
        self.config = config # Keep the whole config for access to image_size, etc.
        self.data_split_name = data_split_name

        if transform is None:
            self.transform = transforms.Compose([
                ResizePIL((config.image_size, config.image_size)),
                ImageToTensor(),
                transforms.Lambda(normalize_image) # Assumes normalize_image is defined in utils
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_tensor = None

        try:
            # Load image
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"ERROR: ImageDataset - Image file not found: {img_path}")
                return None # Indicate failure for this sample
            except Exception as e:
                print(f"ERROR: ImageDataset - Error loading image {img_path}: {e}")
                return None

            # Transform image
            if self.transform:
                try:
                    image_tensor = self.transform(image)
                except Exception as e:
                    print(f"ERROR: ImageDataset - Error applying transform to image {img_path}: {e}")
                    return None
            else: # Should always have a default transform
                image_tensor = image # Fallback, though unlikely if default transform is set

            if image_tensor is None:
                print(f"ERROR: ImageDataset - Image tensor is None for {img_path} after processing. Returning None.")
                return None

            return {
                "image": image_tensor,
                "path": img_path
                # No segments or adj matrix for this simple dataset
            }

        except Exception as e:
            print(f"CRITICAL ERROR in ImageDataset.__getitem__ for {img_path}: {e}. Returning None.")
            return None

# UnpairedImageDataset class removed.
