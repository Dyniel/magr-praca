import argparse
import os
from omegaconf import OmegaConf
from src.data_loader import get_dataloader
# Assuming BaseConfig is used for structuring the config
from configs.base_config import BaseConfig


def test_load(config_path, num_batches_to_test=5):
    # Load base config
    conf = OmegaConf.structured(BaseConfig)

    # Load config from YAML file if provided
    if config_path:
        try:
            file_conf = OmegaConf.load(config_path)
            conf = OmegaConf.merge(conf, file_conf)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config file {config_path}: {e}. Using defaults.")

    print("\nUsing configuration:")
    print(OmegaConf.to_yaml(conf))

    print(f"\nAttempting to get dataloader for 'train' split...")
    # Ensure the dataset path in your config (e.g., conf.dataset_path) is correct for 'train'
    # Or change "train" to "val" or "test" if you want to test those.
    # Also, ensure that the model architecture in the config matches the dataset type you expect to test.
    # For example, if testing ImageToGraphDataset, conf.model.architecture should be "gan6_gat_cnn".

    # To make it more robust for testing, let's check the configured architecture
    # and print a note if it doesn't match common expectations for the dataset types.
    model_arch = getattr(conf.model, "architecture", "gan5_gcn")  # Default to gan5 if not specified
    print(f"Configured model architecture: {model_arch}")
    if model_arch == "gan6_gat_cnn":
        print("Expecting ImageToGraphDataset, which returns (images, graphs) tuples.")
    elif model_arch == "gan5_gcn":
        print("Expecting SuperpixelDataset, which returns dictionaries.")
    else:
        print(f"Unknown model architecture '{model_arch}' for dataset type expectation.")

    train_dataloader = get_dataloader(conf, data_split="train", shuffle=False, drop_last=False)

    if train_dataloader is None:
        print("Failed to create train_dataloader. Exiting.")
        return

    print(f"\nIterating through up to {num_batches_to_test} batches from train_dataloader...")
    batches_processed = 0
    for i, batch in enumerate(train_dataloader):
        if i >= num_batches_to_test:
            break
        batches_processed += 1

        print(f"\n--- Batch {i + 1} ---")
        if batch is None:
            print("Received a None batch from dataloader (collate_fn likely returned None).")
            continue

        # For SuperpixelDataset (gan5)
        if isinstance(batch, dict):
            print(f"  Batch type: dict (expected for SuperpixelDataset)")
            print(
                f"  Image batch shape: {batch['image'].shape if 'image' in batch and batch['image'] is not None else 'N/A'}")
            print(
                f"  Segments batch shape: {batch['segments'].shape if 'segments' in batch and batch['segments'] is not None else 'N/A'}")
            print(f"  Adj batch shape: {batch['adj'].shape if 'adj' in batch and batch['adj'] is not None else 'N/A'}")
            if 'path' in batch and batch['path'] is not None:
                print(
                    f"  First image path in batch: {batch['path'][0] if isinstance(batch['path'], list) and batch['path'] else batch['path']}")

        # For ImageToGraphDataset (gan6)
        elif isinstance(batch, tuple) and len(batch) == 2:
            print(f"  Batch type: tuple (expected for ImageToGraphDataset)")
            images, graphs = batch
            print(f"  Image batch shape: {images.shape if images is not None else 'N/A'}")
            if graphs is not None:
                print(f"  Graphs batch object: {graphs}")  # PyG Batch object
                print(f"  Number of graphs in batch: {graphs.num_graphs if hasattr(graphs, 'num_graphs') else 'N/A'}")
            else:
                print("  Graphs batch component is None.")
        else:
            print(f"  Received batch of unexpected type: {type(batch)}")
            print(f"  Batch content: {str(batch)[:500]}...")  # Print a snippet of the batch

    if batches_processed == 0 and num_batches_to_test > 0:
        print("\nNo batches were processed. The dataloader might be empty or failing early.")
    print(f"\nFinished testing {batches_processed} batches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DataLoader iteration.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/experiment_config.yaml",  # Default to your main experiment config
        help="Path to a YAML configuration file."
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=5,
        help="Number of batches to test."
    )
    args = parser.parse_args()

    # Basic check for config file existence
    if not os.path.exists(args.config_file):
        print(f"ERROR: Config file not found at {args.config_file}")
        print("Please provide a valid path using --config_file")
    else:
        test_load(args.config_file, args.num_batches)
