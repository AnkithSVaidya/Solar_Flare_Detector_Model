import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


# The resolution of the image
RESOLUTION = 128
CUTOFF = 1e-8

def load_gray(path):
    img = Image.open(path).convert("L").resize((RESOLUTION, RESOLUTION))
    return img

def is_black_image(img, threshold=0):
        arr = np.array(img)
        return np.max(arr) <= threshold

def find_folder(root_dir, folder_name):
    for root, dirs, files in os.walk(root_dir):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

class MagnetogramDataset(torch.utils.data.Dataset):

    # Init function 
    def __init__(self, training=True):
        super().__init__()
        # Get the root directory of the path
        self.root_data_dir = find_folder(os.getcwd(), "data")
        self.root_data_dir = os.path.join(self.root_data_dir, 'training' if training else 'test')

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
        self._dir_cache = {}  # reset per-sample; reused across the 10 wavelengths x 4 timesteps below

        images = []

        folder_parts = dataset_id.split("_", 1)
        images_path = os.path.join(self.root_data_dir, folder_parts[0], folder_parts[1])
        all_files = self._get_dir_listing(images_path)
        timestep_images = sorted(f for f in all_files if f.split("__")[1].split(".")[0] == "magnetogram")
        timestep_images.sort(reverse=True)


        current_image = []
        if len(timestep_images) == 0:
            print(f"!!! NO MAGNETOGRAM FILES FOUND for dataset_id={dataset_id}, images_path={images_path}")
        # Get the most recent image in the list
        for curr_img_path in timestep_images:
            curr_img = load_gray(os.path.join(images_path, curr_img_path))
            if not is_black_image(curr_img):
                current_image = curr_img
                break
            print(f"Black image found: {curr_img_path}")
        if current_image is None:
            raise ValueError(f"No valid (non-black) magnetogram found for sample {dataset_id}")
        curr_im_arr = np.array(current_image) / 255.0               # Convert into an array with range [0,1]
        curr_im_arr = curr_im_arr * 2 - 1                           # Rescale to range [-1, 1]

        # import matplotlib.pyplot as plt

        # plt.imshow(curr_im_arr, cmap="seismic", vmin=-1, vmax=1)
        # plt.colorbar(label="Pixel Value")
        # plt.axis("off")
        # plt.show()
        # print(curr_im_arr)

        # Size [128, 128]
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
        metadata = pd.read_csv(os.path.join(self.root_data_dir, "meta_data.csv"))
        metadata["end"] = pd.to_datetime(metadata["end"])
        metadata = metadata.sort_values(by="end")
        metadata = metadata.rename(columns={"end": "datetime"})
        metadata = metadata.drop(columns=['start'])
        metadata["is_flare"] = metadata["peak_flux"] > CUTOFF

        def has_magnetogram(dataset_id):
            folder_parts = dataset_id.split("_", 1)
            images_path = os.path.join(self.root_data_dir, folder_parts[0], folder_parts[1])
            if not os.path.isdir(images_path):
                return False
            files = []
            for _, _, fs in os.walk(images_path):
                files.extend(fs)
            return any(f.split("__")[1].split(".")[0] == "magnetogram" for f in files if "__" in f)

        before = len(metadata)
        metadata = metadata[metadata["id"].apply(has_magnetogram)]
        after = len(metadata)
        if before != after:
            print(f"Dropped {before - after} samples with no magnetogram images ({after} remaining)")

        return metadata
if __name__ == "__main__":
    m = MagnetogramDataset()
    print(m.metadata.head())
