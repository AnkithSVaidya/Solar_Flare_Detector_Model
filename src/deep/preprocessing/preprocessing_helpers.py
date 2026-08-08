# Global Imports
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# Local Imports
from constants import *

# This function augments the image by rotating, shifting, and adding noise
def augment(img, fill_color = 128):
    angle = np.random.uniform(-10, 10)
    im = img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill_color)
    dx, dy = np.random.randint(-3, 4), np.random.randint(-3, 4) 
    im = im.transform((RESOLUTION, RESOLUTION), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=fill_color)
    arr = np.array(im, dtype=np.float32) / 255.0
    arr += np.random.normal(0, 0.02, arr.shape)
    return np.clip(arr, 0, 1)

# This function builds an array of n augmented images
def build(img, label, n):
    X = np.array([augment(img) for _ in range(n)], dtype=np.float32)
    y = np.full((n, 1), label, dtype=np.float32)
    return X, y
     
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
    df = metadata
    
    # Simple random 80/20 split on samples
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        random_state=SEED
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    return train_df, val_df