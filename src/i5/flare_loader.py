

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flare_dataset import FlareDataset

CUTOFF = 1e-8

"""
Class that loads the data (and can be used to perform basic analysis)
"""
class FlareDataLoader():

    def __init__(self):
        # Get the root directory of the path
        self.root_data_dir = find_folder(os.getcwd(), "data")

        # Get the target csv
        self.training_target = self._process_metadata(training=True)
        self.test_target = self._process_metadata(training=False)

        # Generate the flare dataset
        self.training_dataset = FlareDataset(self.training_target, os.path.join(self.root_data_dir, "training"))
        X, y = self.training_dataset[0]


    def _process_metadata(self, training=True):
        direc_to_read = f"{'training' if training else 'test'}/meta_data.csv"

        # Load the target csv
        metadata = pd.read_csv(os.path.join(self.root_data_dir, direc_to_read))

        # Convert date/time column
        metadata["end"] = pd.to_datetime(metadata["end"])

        # Sort the data in terms of ascending datetime
        metadata = metadata.sort_values(by="end")

        # Rename the "id" parameter to "dataset_id"
        metadata = metadata.rename(columns={"id": "dataset_id", "end": "datetime"})

        # Get rid of the "start" column
        metadata = metadata.drop(columns = ['start'])

        # Give each data a unique ID
        metadata["id"] = range(len(metadata))

        metadata["is_flare"] = metadata["peak_flux"] > CUTOFF

        # Set the index for the data
        metadata = metadata.set_index("id")

        return metadata

    def show_flux_histogram(self, training=True):
        curr_target = self.training_target.copy() if training else self.test_target.copy()
        plt.hist(np.log10(curr_target["peak_flux"]), bins=50)

        # Plot cutoff
        plt.axvline(np.log10(CUTOFF), linestyle="--", color='r', label="Flare Threshold")

        plt.xlabel("Peak Flux")
        plt.ylabel("Count")
        plt.title(f"Solar Flare Peak Flux Distribution in {'Training' if training else 'Test'} Data")
        plt.show()
    def show_class_distribution(self, training=True):
        """
        Display the number of flare and non-flare samples.
        """

        curr_target = (
            self.training_target.copy()
            if training
            else self.test_target.copy()
        )

        counts = curr_target["is_flare"].value_counts()

        print(f"{'Training' if training else 'Test'} Class Distribution:")
        print(f"Non-flares: {counts.get(False, 0)}")
        print(f"Flares: {counts.get(True, 0)}")

        plt.bar(
            ["Non-Flare", "Flare"],
            [
                counts.get(False, 0),
                counts.get(True, 0)
            ]
        )

        plt.ylabel("Count")
        plt.title(
            f"{'Training' if training else 'Test'} Flare Distribution"
        )
        plt.show()

def find_folder(root_dir, folder_name):
    for root, dirs, files in os.walk(root_dir):
        if folder_name in dirs:
            return os.path.join(root, folder_name)
    return None

if __name__ == "__main__":
    flareDataLoader = FlareDataLoader()
    flareDataLoader.show_class_distribution()
# def load_targets():
#     data_dir = find_folder("data/training", str(sample_id))

#     metadata_path = os.path.join(os.path.dirname(data_dir), "meta_data.csv")
#     if not os.path.exists(metadata_path):
#         return [], [], []
#     metadata = pd.read_csv(metadata_path)





# # img1 = load_gray('data\\training\\11388\\2012_01_07_02_27_01_0\\2012-01-06T142701__94.jpg')
# # display(img1)