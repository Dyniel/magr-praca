import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms # Will be used for basic transforms if not passed custom ones

from src.utils import (
    precompute_superpixels_for_dataset, get_image_paths,
    ImageToTensor, ResizePIL, normalize_image,
    convert_image_to_pyg_graph, PYG_AVAILABLE # For gan6
)

if PYG_AVAILABLE:
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGData
else:
    PyGBatch = None # Placeholder
    PyGData = None


class SuperpixelDataset(Dataset): # For gan5-style models
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


# --- For gan6-style models using PyTorch Geometric ---

class ImageToGraphDataset(Dataset):
    def __init__(self, image_paths, config, image_transform=None):
        """
        Dataset for gan6-style models. Converts images to PyG graph objects.
        Args:
            image_paths (list): List of paths to images.
            config (object): Configuration object.
            image_transform (callable, optional): Transform for the real image tensor.
        """
        if not PYG_AVAILABLE or PyGData is None:
            raise ImportError("PyTorch Geometric is required for ImageToGraphDataset.")

        self.image_paths = image_paths
        self.config = config

        # Transform for the image itself (e.g., to tensor, normalize for D)
        if image_transform is None:
            self.image_transform = transforms.Compose([
                ResizePIL((config.image_size, config.image_size)), # Resize before graph conversion too
                ImageToTensor(), # Converts to [0,1] tensor C,H,W
                transforms.Lambda(normalize_image) # Normalizes to [-1,1]
            ])
        else:
            self.image_transform = image_transform

        # Cache directory for PyG Data objects
        # Example: cache_root/pyg_graphs_sp100_slic10_is256/
        self.graph_cache_dir = os.path.join(
            config.cache_dir,
            f"pyg_graphs_sp{config.model.gan6_num_superpixels}_slic{config.model.gan6_slic_compactness}_is{config.image_size}"
        )
        self._prepare_graph_cache()

    def _prepare_graph_cache(self):
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        print(f"Using/creating PyG graph cache at: {self.graph_cache_dir}")

        missing_cache_files = False
        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cache_file = os.path.join(self.graph_cache_dir, f"{base_name}.pt")
            if not os.path.exists(cache_file):
                missing_cache_files = True
                break

        if missing_cache_files:
            print("Preprocessing images to PyG graphs for caching...")
            for img_path in tqdm(self.image_paths, desc="Caching PyG Graphs"):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cache_file = os.path.join(self.graph_cache_dir, f"{base_name}.pt")
                if os.path.exists(cache_file):
                    continue

                try:
                    # Load image, resize, convert to numpy [0,1] for graph conversion
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_resized = pil_img.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
                    img_np_01 = np.array(pil_resized).astype(np.float32) / 255.0
                    if img_np_01.ndim == 2: # Grayscale
                         img_np_01 = np.expand_dims(img_np_01, axis=-1)
                    if img_np_01.shape[-1] != 3 and img_np_01.shape[-1] == 1 : # If grayscale with one channel, make it 3 for consistency
                        img_np_01 = np.concatenate([img_np_01]*3, axis=-1)


                    graph_data = convert_image_to_pyg_graph(
                        img_np_01,
                        num_superpixels=self.config.model.gan6_num_superpixels,
                        slic_compactness=self.config.model.gan6_slic_compactness
                        # feature_type could be added to config if more options are needed
                    )
                    torch.save(graph_data, cache_file)
                except Exception as e:
                    print(f"Error processing and caching graph for {img_path}: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        graph_cache_file = os.path.join(self.graph_cache_dir, f"{base_name}.pt")

        try:
            graph_data = torch.load(graph_cache_file)
        except Exception as e:
            # This should ideally be caught by _prepare_graph_cache
            raise RuntimeError(f"Error loading cached graph data for {img_path} from {graph_cache_file}: {e}")

        # Load and transform the real image (for Discriminator input)
        pil_img = Image.open(img_path).convert("RGB")
        real_image_tensor = self.image_transform(pil_img)

        return real_image_tensor, graph_data


def collate_graphs(batch):
    """Collate function for ImageToGraphDataset."""
    if not PYG_AVAILABLE or PyGBatch is None:
        raise ImportError("PyTorch Geometric is required for collate_graphs.")

    real_images, graph_data_objects = zip(*batch)
    # Filter out None graphs if any failed during loading (should not happen with proper caching)
    valid_graphs = [g for g in graph_data_objects if g is not None]
    if len(valid_graphs) != len(graph_data_objects):
        print(f"Warning: Some graph data objects were None. Found {len(valid_graphs)} valid graphs out of {len(graph_data_objects)}.")
        # This might lead to batch size mismatch if not handled carefully.
        # For now, assume all graphs are valid due to caching.

    return torch.stack(real_images), PyGBatch.from_data_list(valid_graphs)


def get_dataloader(config, shuffle=True):
    """
    Creates and returns a DataLoader.
    Selects dataset type based on config.model.architecture.
    """
    image_paths = get_image_paths(config.dataset_path)
    if not image_paths:
        raise ValueError(f"No images found in {config.dataset_path}")

    if config.debug_num_images > 0 and config.debug_num_images < len(image_paths):
        print(f"Using a subset of {config.debug_num_images} images for debugging.")
        image_paths = image_paths[:config.debug_num_images]

    dataset_type = getattr(config.model, "architecture", "gan5_gcn") # Default to gan5

    if dataset_type == "gan6_gat_cnn":
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for 'gan6_gat_cnn' architecture but not found.")
        print("Using ImageToGraphDataset for gan6_gat_cnn architecture.")
        dataset = ImageToGraphDataset(image_paths=image_paths, config=config)
        collate_fn_to_use = collate_graphs
    elif dataset_type == "gan5_gcn":
        print("Using SuperpixelDataset for gan5_gcn architecture.")
        dataset = SuperpixelDataset(image_paths=image_paths, config=config)
        collate_fn_to_use = None # Use default collate for SuperpixelDataset
    else:
        raise ValueError(f"Unsupported model.architecture: {dataset_type}")

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_to_use
    )
    return dataloader

print("src/data_loader.py created and populated with SuperpixelDataset and get_dataloader.")
