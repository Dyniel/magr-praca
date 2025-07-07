import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms # Will be used for basic transforms if not passed custom ones
from tqdm import tqdm

from src.utils import (
    precompute_superpixels_for_dataset, get_image_paths,
    ImageToTensor, ResizePIL, normalize_image,
    convert_image_to_pyg_graph, PYG_AVAILABLE
)

if PYG_AVAILABLE:
    from torch_geometric.data import Batch as PyGBatch
    from torch_geometric.data import Data as PyGData
else:
    PyGBatch = None
    PyGData = None


class SuperpixelDataset(Dataset): # For gan5-style models
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

        expected_compactness = getattr(config, 'slic_compactness', 10.0) # Default if not in config

        if not os.path.exists(self.cache_dir) or not os.listdir(self.cache_dir):
            print(f"Cache directory {self.cache_dir} for split '{data_split_name}' not found or empty. Precomputing superpixels...")
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
            for img_path in self.image_paths: # Ensure image_paths are populated before this loop
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
                if data_split_name == "train": current_dataset_path = getattr(config, 'dataset_path', None)
                elif data_split_name == "val": current_dataset_path = getattr(config, 'dataset_path_val', None)
                elif data_split_name == "test": current_dataset_path = getattr(config, 'dataset_path_test', None)

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

        try:
            data = np.load(cache_file_path)
            segments = torch.from_numpy(data["segments"]).long()
            adj_matrix = torch.from_numpy(data["adj"]).float()
        except FileNotFoundError:
            raise FileNotFoundError(f"Cache file not found for {img_path} at {cache_file_path}. ")
        except Exception as e:
            raise RuntimeError(f"Error loading cached data for {img_path}: {e}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

        if self.transform:
            image_tensor = self.transform(image)

        if self.target_transform: # Should not be used typically
            segments = self.target_transform(segments)
            adj_matrix = self.target_transform(adj_matrix)

        return {
            "image": image_tensor,
            "segments": segments,
            "adj": adj_matrix,
            "path": img_path
        }

class ImageToGraphDataset(Dataset):
    def __init__(self, image_paths, config, data_split_name="train", image_transform=None):
        if not PYG_AVAILABLE or PyGData is None:
            raise ImportError("PyTorch Geometric is required for ImageToGraphDataset.")

        self.image_paths = image_paths
        self.config = config
        self.data_split_name = data_split_name

        if image_transform is None:
            self.image_transform = transforms.Compose([
                ResizePIL((config.image_size, config.image_size)),
                ImageToTensor(),
                transforms.Lambda(normalize_image)
            ])
        else:
            self.image_transform = image_transform

        self.graph_cache_dir = os.path.join(
            config.cache_dir,
            f"{self.data_split_name}_pyg_graphs_sp{config.model.gan6_num_superpixels}_slic{config.model.gan6_slic_compactness}_is{config.image_size}"
        )
        self._prepare_graph_cache()

    def _prepare_graph_cache(self):
        os.makedirs(self.graph_cache_dir, exist_ok=True)
        print(f"Using/creating PyG graph cache for {self.data_split_name} at: {self.graph_cache_dir}")

        missing_cache_files = False
        # Ensure image_paths are populated
        if not self.image_paths:
            current_dataset_path = None
            if self.data_split_name == "train": current_dataset_path = getattr(self.config, 'dataset_path', None)
            elif self.data_split_name == "val": current_dataset_path = getattr(self.config, 'dataset_path_val', None)
            elif self.data_split_name == "test": current_dataset_path = getattr(self.config, 'dataset_path_test', None)
            if current_dataset_path:
                self.image_paths = get_image_paths(current_dataset_path)
            if not self.image_paths: # If still no paths, can't proceed
                 print(f"Warning: No image paths found for split '{self.data_split_name}' during graph cache preparation.")
                 return # Or raise error, but returning allows dataloader to be None later

        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cache_file = os.path.join(self.graph_cache_dir, f"{base_name}.pt")
            if not os.path.exists(cache_file):
                missing_cache_files = True
                break

        if missing_cache_files:
            print(f"Preprocessing images to PyG graphs for caching (split: {self.data_split_name})...")
            if not self.image_paths: # Should be populated above, but double check
                print(f"Error: No image paths to process for graph caching for split '{self.data_split_name}'.")
                return

            for img_path in tqdm(self.image_paths, desc=f"Caching PyG Graphs ({self.data_split_name})"):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                cache_file = os.path.join(self.graph_cache_dir, f"{base_name}.pt")
                if os.path.exists(cache_file):
                    continue
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_resized = pil_img.resize((self.config.image_size, self.config.image_size), Image.BILINEAR) # TODO: check interpolation consistency
                    img_np_01 = np.array(pil_resized).astype(np.float32) / 255.0
                    if img_np_01.ndim == 2:
                         img_np_01 = np.expand_dims(img_np_01, axis=-1)
                    if img_np_01.shape[-1] != 3 and img_np_01.shape[-1] == 1 :
                        img_np_01 = np.concatenate([img_np_01]*3, axis=-1)

                    graph_data = convert_image_to_pyg_graph(
                        img_np_01,
                        num_superpixels=self.config.model.gan6_num_superpixels,
                        slic_compactness=self.config.model.gan6_slic_compactness
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
            raise RuntimeError(f"Error loading cached graph data for {img_path} from {graph_cache_file}: {e}")

        pil_img = Image.open(img_path).convert("RGB")
        real_image_tensor = self.image_transform(pil_img)

        return real_image_tensor, graph_data


def collate_graphs(batch):
    if not PYG_AVAILABLE or PyGBatch is None:
        raise ImportError("PyTorch Geometric is required for collate_graphs.")

    # Filter out None items that could result from __getitem__ failing before returning a tuple
    # This can happen if an image is corrupted or a cache file is bad for a single item.
    # However, __getitem__ in this class raises RuntimeError on load failure, so batch should be clean.
    # But if an image_path was problematic and _prepare_graph_cache skipped it, it might lead here.
    # For robustness, filter None items from batch if they can occur.
    # batch = [item for item in batch if item is not None and item[1] is not None]
    # if not batch: return None # If all items failed

    real_images, graph_data_objects = zip(*batch)
    valid_graphs = [g for g in graph_data_objects if g is not None]

    if len(valid_graphs) != len(graph_data_objects):
        print(f"Warning: Some graph data objects were None. Found {len(valid_graphs)} valid graphs out of {len(graph_data_objects)}.")
        if not valid_graphs: # All graphs were None
             return None # Or handle differently, e.g. return empty tensors / special batch

    # If after filtering, valid_graphs is empty but real_images might not be (if some graphs failed)
    # This would cause PyGBatch.from_data_list to fail if valid_graphs is empty.
    if not valid_graphs:
        # This case should ideally be handled by ensuring all data is processable or by
        # more sophisticated error handling in __getitem__ or _prepare_graph_cache.
        # For now, if no valid graphs, we can't form a batch.
        # Consider returning None or an empty structure that the training loop can skip.
        print("Warning: No valid graphs in batch after filtering. Skipping batch.")
        return None # This will need to be handled by the training loop

    return torch.stack(real_images), PyGBatch.from_data_list(valid_graphs)


def get_dataloader(config, data_split="train", shuffle=True, drop_last=True):
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
    dataset = None
    collate_fn_to_use = None

    if dataset_type == "gan6_gat_cnn":
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for 'gan6_gat_cnn' architecture but not found.")
        print(f"Using ImageToGraphDataset for {data_split} (gan6_gat_cnn architecture).")
        dataset = ImageToGraphDataset(image_paths=image_paths, config=config, data_split_name=data_split)
        collate_fn_to_use = collate_graphs
    elif dataset_type == "gan5_gcn":
        print(f"Using SuperpixelDataset for {data_split} (gan5_gcn architecture).")
        dataset = SuperpixelDataset(image_paths=image_paths, config=config, data_split_name=data_split)
        # Default collate_fn is fine for SuperpixelDataset
    else:
        raise ValueError(f"Unsupported model.architecture: {dataset_type}")

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
# Removed the print("src/data_loader.py created...") as it's an overwrite
