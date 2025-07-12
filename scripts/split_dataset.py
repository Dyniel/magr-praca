import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import glob

def split_dataset(source_dir, dest_dir, train_size, val_size, test_size, random_state=42):
    """
    Splits a directory of images into training, validation, and test sets.

    Args:
        source_dir (str): The directory containing the images.
        dest_dir (str): The directory where the 'train', 'val', and 'test' subdirectories will be created.
        train_size (float): The proportion of the dataset to allocate to the training set.
        val_size (float): The proportion of the dataset to allocate to the validation set.
        test_size (float): The proportion of the dataset to allocate to the test set.
        random_state (int): The seed for the random number generator.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    image_paths = glob.glob(os.path.join(source_dir, '*.png')) + \
                  glob.glob(os.path.join(source_dir, '*.jpg')) + \
                  glob.glob(os.path.join(source_dir, '*.jpeg'))

    if not image_paths:
        print(f"No images found in {source_dir}")
        return

    # First split to separate out the test set
    train_val_paths, test_paths = train_test_split(image_paths, test_size=test_size, random_state=random_state)

    # Adjust val_size to be a proportion of the remaining train_val set
    val_proportion_of_train_val = val_size / (train_size + val_size)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_proportion_of_train_val, random_state=random_state)

    def copy_files(paths, dest):
        for path in paths:
            shutil.copy(path, dest)

    copy_files(train_paths, train_dir)
    copy_files(val_paths, val_dir)
    copy_files(test_paths, test_dir)

    print(f"Copied {len(train_paths)} images to {train_dir}")
    print(f"Copied {len(val_paths)} images to {val_dir}")
    print(f"Copied {len(test_paths)} images to {test_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train, val, and test sets.')
    parser.add_argument('--source_dir', type=str, required=True, help='Source directory with images.')
    parser.add_argument('--dest_dir', type=str, required=True, help='Destination directory for splits.')
    parser.add_argument('--train_size', type=float, default=0.7, help='Training set size.')
    parser.add_argument('--val_size', type=float, default=0.15, help='Validation set size.')
    parser.add_argument('--test_size', type=float, default=0.15, help='Test set size.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility.')

    args = parser.parse_args()

    if args.train_size + args.val_size + args.test_size > 1.0:
        raise ValueError("The sum of train, val, and test sizes cannot be greater than 1.")

    split_dataset(args.source_dir, args.dest_dir, args.train_size, args.val_size, args.test_size, args.random_state)
