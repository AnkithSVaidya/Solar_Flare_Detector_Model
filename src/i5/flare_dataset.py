import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os

RESOLUTION = 128
WAVELENGTHS = ['94', '131', '171', '193', '211', '304', '335', '1700', 'continuum', 'magnetogram']

def is_black_image(img, threshold=0):
        arr = np.array(img)
        return np.max(arr) <= threshold

class FlareDataset(torch.utils.data.Dataset):

    # Init function 
    def __init__(self, metadata, data_dir):
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Grayscale(), transforms.Resize((RESOLUTION, RESOLUTION)), transforms.ToTensor()])

    # Function that returns the length
    def __len__(self):
        return len(self.metadata)

    # Get an item based on index
    def __getitem__(self, idx):

        # Get the sample from the metadata
        row = self.metadata.iloc[idx]
        dataset_id = row["dataset_id"]
        self._dir_cache = {}  # reset per-sample; reused across the 10 wavelengths x 4 timesteps below

        images = []

        for timestep in range(4):
            channels = []
            masks = []


            for wavelength in WAVELENGTHS:
                img, valid = self.load_image(dataset_id, timestep, wavelength)
                channels.append(img)
                masks.append(valid)

            # Size [20, 128, 128]
            timestep_imgs = torch.stack(channels)
            timestep_masks = torch.stack(masks)
            images.append(torch.cat([timestep_imgs, timestep_masks], dim=0))
            
        # Size [4, 10, 128, 128]
        X = torch.stack(images)
        y = torch.tensor(row["is_flare"], dtype=torch.float32)

        return X, y
    
    def load_image(self, dataset_id, timestep, wavelength):
        # Get all of the images at the right wavelength
        folder_parts = dataset_id.split("_", 1)
        images_path = os.path.join(self.data_dir, folder_parts[0], folder_parts[1])

        all_files = self._get_dir_listing(images_path)
        timestep_images = sorted(f for f in all_files if f.split("__")[1].split(".")[0] == wavelength)

        if len(timestep_images) <= timestep:
            return torch.zeros(RESOLUTION, RESOLUTION), torch.zeros(RESOLUTION, RESOLUTION)

        path = os.path.join(
            images_path,
            timestep_images[timestep]
        )
        path = os.path.join(images_path, timestep_images[timestep])

        # Get the Image from the path
        img = Image.open(path)
        img_transformed = self.transform(img)


        # Check if the image is all black
        if is_black_image(img):
            return img_transformed.squeeze(0), torch.zeros(RESOLUTION, RESOLUTION)

        return img_transformed.squeeze(0), torch.ones(RESOLUTION, RESOLUTION)
    
    def _get_dir_listing(self, images_path):
        if images_path not in self._dir_cache:
            files = []
            for root, dirs, fs in os.walk(images_path):
                files.extend(fs)
            self._dir_cache[images_path] = files
        return self._dir_cache[images_path]
        