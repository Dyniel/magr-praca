import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms # Will be used for basic transforms if not passed custom ones

from src.utils import precompute_superpixels_for_dataset, get_image_paths, ImageToTensor, ResizePIL, normalize_image

class SuperpixelDataset(Dataset):
    def __init__(self, image_paths, config, transform=None, target_transform=None):
        """
        Args:
            image_paths (list): List of paths to images.
            config (object): Configuration object with attributes like image_size,
                             num_superpixels, cache_dir, dataset_path.
            transform (callable, optional): Optional transform to be applied on a sample image.
            target_transform (callable, optional): Optional transform to be applied on segmentation/adj.
                                                 (Less common for this setup).
        """
        self.image_paths = image_paths
        self.config = config
        self.cache_dir = os.path.join(config.cache_dir, f"sp_{config.num_superpixels}_is_{config.image_size}")

        # Define default transforms if none are provided
        if transform is None:
            self.transform = transforms.Compose([
                ResizePIL((config.image_size, config.image_size)),
                ImageToTensor(), # Converts to [0,1] tensor C,H,W
                transforms.Lambda(normalize_image) # Normalizes to [-1,1]
            ])
        else:
            self.transform = transform

        self.target_transform = target_transform # Usually not needed for seg/adj

        # Precompute superpixels if cache is not valid or not present
        # A simple check: does the cache directory exist and is not empty?
        # For more robust caching, one might check if all files exist or use a manifest.
        if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
            print(f"Cache directory {self.cache_dir} not found or empty. Precomputing superpixels...")
            # Ensure image_paths are from the configured dataset_path if self.image_paths is empty
            # or if we want to strictly use paths from the config.
            # For now, assume image_paths provided to constructor are the ones to use.
            if not self.image_paths:
                 print("Warning: image_paths is empty. Attempting to get paths from config.dataset_path.")
                 self.image_paths = get_image_paths(config.dataset_path)

            if not self.image_paths:
                raise ValueError("No image paths provided or found in config.dataset_path for precomputation.")

            precompute_superpixels_for_dataset(
                image_paths=self.image_paths,
                cache_dir=self.cache_dir,
                image_size=config.image_size,
                num_superpixels_config=config.num_superpixels,
                compactness=config.slic_compactness # Assuming slic_compactness is in config
            )
        else:
            print(f"Using existing cache directory: {self.cache_dir}")
            # Verify all expected cache files exist for self.image_paths
            missing_files = False
            for img_path in self.image_paths:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cache_file_path = os.path.join(self.cache_dir, base_name + ".npz")
                if not os.path.exists(cache_file_path):
                    print(f"Cache file missing for {img_path} at {cache_file_path}")
                    missing_files = True
                    break
            if missing_files:
                print("Some cache files are missing. Re-running precomputation.")
                if not self.image_paths:
                     self.image_paths = get_image_paths(config.dataset_path)
                precompute_superpixels_for_dataset(
                    image_paths=self.image_paths,
                    cache_dir=self.cache_dir,
                    image_size=config.image_size,
                    num_superpixels_config=config.num_superpixels,
                    compactness=config.slic_compactness
                )


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_file_path = os.path.join(self.cache_dir, base_name + ".npz")

        try:
            data = np.load(cache_file_path)
            segments = torch.from_numpy(data["segments"]).long()  # Ensure it's LongTensor for F.one_hot
            adj_matrix = torch.from_numpy(data["adj"]).float()
        except FileNotFoundError:
            # This should ideally be caught by the __init__ precomputation logic
            raise FileNotFoundError(f"Cache file not found for {img_path} at {cache_file_path}. "
                                    "Please ensure precomputation ran correctly.")
        except Exception as e:
            raise RuntimeError(f"Error loading cached data for {img_path}: {e}")

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        # Apply transforms
        if self.transform:
            image_tensor = self.transform(image)

        # Target transforms are less common here but included for completeness
        if self.target_transform:
            segments = self.target_transform(segments)
            adj_matrix = self.target_transform(adj_matrix)

        return {
            "image": image_tensor,      # The real image, transformed
            "segments": segments,       # Segmentation mask for the image
            "adj": adj_matrix,          # Adjacency matrix for the superpixels
            "path": img_path            # Path to the original image
        }

def get_dataloader(config, shuffle=True):
    """
    Creates and returns a DataLoader for the SuperpixelDataset.
    """
    image_paths = get_image_paths(config.dataset_path)
    if not image_paths:
        raise ValueError(f"No images found in {config.dataset_path}")

    if config.debug_num_images > 0:
        print(f"Using a subset of {config.debug_num_images} images for debugging.")
        image_paths = image_paths[:config.debug_num_images]

    dataset = SuperpixelDataset(image_paths=image_paths, config=config)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True, # Good practice if using CUDA
        drop_last=True # Important for some GAN training if batch consistency is needed
    )
    return dataloader

print("src/data_loader.py created and populated with SuperpixelDataset and get_dataloader.")
