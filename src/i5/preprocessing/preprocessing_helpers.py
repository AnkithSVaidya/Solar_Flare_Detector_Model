# Global Imports
from PIL import Image
import os
import numpy as np

# Local Imports
from constants import *

# This function loads an image and resizes it
def load_gray(path):
    img = Image.open(path).convert("L").resize((RESOLUTION, RESOLUTION))
    return img

# This function checks if an image is all black
# The threshold is what it considers 'noise'
def is_black_image(img, threshold=0):
        arr = np.array(img)
        return np.max(arr) <= threshold

# This function finds a folder in a directory
def find_folder(root_dir, folder_name):
    for root, dirs, files in os.walk(root_dir):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

def split_metadata(metadata):
    # Copy the metadata & split into different regions
    metadata = metadata.copy()
    metadata["region"] = metadata["id"].apply(lambda d: d.split("_", 1)[0])

    # For each region, get its most severe label (max is_flare, or better: max peak_flux)
    region_severity = metadata.groupby("region")["peak_flux"].max()

    # Bucket into quiet vs flaring regions (or finer buckets if you want GOES-class-like granularity)
    flaring_regions = region_severity[region_severity > 1e-8].index.tolist()
    quiet_regions = region_severity[region_severity <= 1e-8].index.tolist()

    # Shuffle the flaring data & non-flaring data to maintain percentagess
    rng = np.random.default_rng(SEED)
    rng.shuffle(flaring_regions)
    rng.shuffle(quiet_regions)

    # Split the flaring & quiet data into validation & training data
    val_flaring = set(flaring_regions[:int(len(flaring_regions) * VALIDATION_SPLIT)])
    val_quiet = set(quiet_regions[:int(len(quiet_regions) * VALIDATION_SPLIT)])
    val_regions = val_flaring | val_quiet

    # Combine the data into train & validation sets
    is_val = metadata["region"].isin(val_regions)
    train_metadata = metadata[~is_val].drop(columns=["region"])
    val_metadata = metadata[is_val].drop(columns=["region"])

    return train_metadata, val_metadata