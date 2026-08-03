import torch
import numpy as np
import pandas as pd
import os

# Local imports
from .preprocessing_helpers import *

class FlareDataset(torch.utils.data.Dataset):
    """
    This is a class that can load data from any source
    """

    def __init__(self, features, type="train"):
        """Default Initialization function
        
        Parameters
        ----------
        features : list[string]
            List of which features to keep
        validation : string
            Either train, val, or test
        """

        # Super Initialization
        super().__init__()

        # Store the relevant input parameters
        self._dir_cache = {}
        self.features = features
        self.type = type

        # Get the root directory of the path
        self.root_data_dir = find_folder(os.getcwd(), "data")
        self.root_data_dir = os.path.join(self.root_data_dir, 'test' if type == 'test' else 'training')


        # Get the target csv
        self.metadata = self._process_metadata()

    def _process_metadata(self):
        """ 
        Processes the metadata & returns a trimmed array 
        """

        # Read the metadata from the csv file
        metadata = pd.read_csv(os.path.join(self.root_data_dir, "meta_data.csv"))

        # Convert the end date to datetime
        metadata["end"] = pd.to_datetime(metadata["end"])

        # Sort the values by the end date
        metadata = metadata.sort_values(by="end")

        # Rename the 'end' to 'datetime' and drop the start (it can be assumed)
        metadata = metadata.rename(columns={"end": "datetime"})
        metadata = metadata.drop(columns=['start'])

        # Add the boolean operator
        metadata["is_flare"] = metadata["peak_flux"] > CUTOFF

        # Check if each dataset used actually has all the features
        before = len(metadata)
        metadata = metadata[metadata["id"].apply(self._has_all_relevant_data)]
        after = len(metadata)

        # Check if we are a test
        if self.type == "test":
            return metadata
        else:
            train, val = split_metadata(metadata)
            return val if self.type == "val" else train

    def _get_relevant_path(self, dataset_id):
        """
        This gets the folder that has the images based on the dataset id
        """

        # Split the ID into the first and second part
        folder_parts = dataset_id.split("_", 1)

        # Get the path to the images folder
        image_folder_path = os.path.join(self.root_data_dir, folder_parts[0], folder_parts[1])

        return image_folder_path
    
    def _has_all_relevant_data(self, dataset_id):
        """
        This class checks if the row has all the data needed (nothing missing)
        """
        # The list of relevant files
        files = []

        # Go through each file in the folder and add it to the list
        for _, _, fs in os.walk(self._get_relevant_path(dataset_id)):
            files.extend(fs)

        # Check if each relevant type is missing
        for feature in self.features:
            if not any(f.split("__")[1].split(".")[0] == feature for f in files if "__" in f):
                return False

        # We have all the relevant types!
        return True

    # Function that returns the length
    def __len__(self):
        return len(self.metadata)

    # Get an item based on index
    def __getitem__(self, idx):

        # Get the sample from the metadata
        row = self.metadata.iloc[idx]
        dataset_id = row["id"]
        image_file_dir = self._get_relevant_path(dataset_id)
        
        # Get all of the files in that images folder
        all_files = self._get_dir_listing(image_file_dir)

        # Initialize a list of images
        images = []

        # Go through each feature and get the most recent image
        for feature in self.features:

            # Get every image that applies to this feature
            timestep_images = sorted(f for f in all_files if f.split("__")[1].split(".")[0] == feature)

            # Reverse-sort the images
            timestep_images.sort(reverse=True)

            # Get the most recent image that isn't black
            for curr_img_path in timestep_images:

                # Get the path of the most recent path
                curr_img = load_gray(os.path.join(image_file_dir, curr_img_path))

                # Check if the image is black. If so, discard
                if not is_black_image(curr_img):
                    images.append(curr_img)
                    break


        # Go through each image and normalize them
        for i in range(len(self.features)):

            # Normalize the image
            images[i] = np.array(images[i]) / 255.0

            # Do some unique things for certain features
            if self.features[i] == 'magnetogram':
                images[i] = images[i] * 2 - 1

            elif self.features[i] == 'continuum':
                images[i] = 1 - images[i]

        # Convert into torch tensors & return
        images = np.array(images)
        X = torch.tensor(images).float()
        y = torch.tensor(row["is_flare"], dtype=torch.float32)
        return X, y

    def _get_dir_listing(self, images_path):
        if images_path not in self._dir_cache:
            self._dir_cache[images_path] = os.listdir(images_path)
        return self._dir_cache[images_path]


    def _resolve_files_for_sample(self, dataset_id):
        """Runs ONCE per sample at dataset construction, not every epoch."""
        image_file_dir = self._get_relevant_path(dataset_id)
        all_files = self._get_dir_listing(image_file_dir)
        resolved = {}
        for feature in self.features:
            timestep_images = sorted(
                (f for f in all_files if f.split("__")[1].split(".")[0] == feature),
                reverse=True
            )
            for f in timestep_images:
                img = load_gray(os.path.join(image_file_dir, f))
                if not is_black_image(img):
                    resolved[feature] = f
                    break
        return resolved