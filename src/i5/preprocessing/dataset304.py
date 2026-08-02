import torch
import numpy as np
import pandas as pd
import os

from .preprocessing_helpers import *

class Dataset304(torch.utils.data.Dataset):

    def __init__(self, training=True, validation=False):
        """Default Initialization function

        Parameters
        ----------
        training : bool
            Whether to load from the training/ folder or the test/ folder
        validation : bool
            If training = True, validation tells whether to use the training split or val split
        """
        super().__init__()

        # Get the root directory of the path
        self.root_data_dir = find_folder(os.getcwd(), "data")
        self.root_data_dir = os.path.join(self.root_data_dir, 'training' if training else 'test')

        self.is_validation = validation

        # Get the target csv
        self.metadata = self._process_metadata()
        

    # Function that returns the length
    def __len__(self):
        return len(self.metadata)

    # Get an item based on index
    def __getitem__(self, idx):

        # Get the sample from the metadata
        row = self.metadata.iloc[idx]
        dataset_id = row["id"]

        # Split the id into parts (useful for finding the image)
        folder_parts = dataset_id.split("_", 1)

        # Get the root path that holds the images
        images_path = os.path.join(self.root_data_dir, folder_parts[0], folder_parts[1])

        # Create a blank cache of images
        self._dir_cache = {}
        
        # Get all of the files in that images folder
        all_files = self._get_dir_listing(images_path)

        # Get the timestep images (specifically the 304 ones)
        timestep_images = sorted(f for f in all_files if f.split("__")[1].split(".")[0] == "304")
        timestep_images.sort(reverse=True)

        # Check if no 304 images were founds
        current_image = []
        if len(timestep_images) == 0:
            print(f"!!! NO 304 FILES FOUND for dataset_id={dataset_id}, images_path={images_path}")

        # Get the most recent image in the list
        for curr_img_path in timestep_images:
            curr_img = load_gray(os.path.join(images_path, curr_img_path))

            # Check if the image is black (if so, discard)
            if not is_black_image(curr_img):
                current_image = curr_img
                break
            print(f"Black image found: {curr_img_path}")

        # If there is no image, raise error
        if current_image is None:
            raise ValueError(f"No valid (non-black) 304 found for sample {dataset_id}")

        # Normalize image pixels
        curr_im_arr = np.array(current_image) / 255.0               # Convert into an array with range [0,1]
        curr_im_arr = curr_im_arr * 2 - 1                           # Rescale to range [-1, 1]

        # Convert into torch tensors & return
        X = torch.from_numpy(curr_im_arr).float()
        y = torch.tensor(row["is_flare"], dtype=torch.float32)
        return X, y

    
    def _get_dir_listing(self, images_path):
        if images_path not in self._dir_cache:
            files = []
            for root, dirs, fs in os.walk(images_path):
                files.extend(fs)
            self._dir_cache[images_path] = files
        return self._dir_cache[images_path]

    def _process_metadata(self):
        # Read the metadata from the csv file
        metadata = pd.read_csv(os.path.join(self.root_data_dir, "meta_data.csv"))

        # Convert the end date to datetime
        metadata["end"] = pd.to_datetime(metadata["end"])
        metadata = metadata.sort_values(by="end")
        metadata = metadata.rename(columns={"end": "datetime"})
        metadata = metadata.drop(columns=['start'])
        metadata["is_flare"] = metadata["peak_flux"] > CUTOFF

        # Check if each dataset used actually has a 304
        def has_magnetogram(dataset_id):
            folder_parts = dataset_id.split("_", 1)
            images_path = os.path.join(self.root_data_dir, folder_parts[0], folder_parts[1])
            if not os.path.isdir(images_path):
                return False
            files = []
            for _, _, fs in os.walk(images_path):
                files.extend(fs)
            return any(f.split("__")[1].split(".")[0] == "304" for f in files if "__" in f)

        before = len(metadata)
        metadata = metadata[metadata["id"].apply(has_magnetogram)]
        after = len(metadata)
        if before != after:
            print(f"Dropped {before - after} samples with no 304 images ({after} remaining)")

        train, val = split_metadata(metadata)
        return val if self.is_validation else train

    
    
if __name__ == "__main__":
    m = MagnetogramDataset()
    print(m.metadata.head())
