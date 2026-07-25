import os
import kagglehub
from PIL import Image

IMG = 64

# Download the dataset
dataset_path = kagglehub.dataset_download("fhnw-i4ds/sdobenchmark")

print("Dataset downloaded to:")
print(dataset_path)


def load_gray(path):
    img = Image.open(path).convert("L").resize((IMG, IMG))
    return img


# Path to the 11388 folder
folder_path = os.path.join(
    dataset_path,
    "training",
    "11388"
)

print("Folder path:", folder_path)


# Find all images inside the folder and subfolders
image_paths = []

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_paths.append(os.path.join(root, file))


print("Number of images found:", len(image_paths))


# Load the first image
if len(image_paths) > 0:
    img1 = load_gray(image_paths[0])

    print("First image:", image_paths[0])
    print("Image size:", img1.size)

    display(img1)