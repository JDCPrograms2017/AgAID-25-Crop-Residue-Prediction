import torch
import os
import numpy as np
import multiprocessing
from torchvision import datasets, transforms # We want to use the ToTensor() transform.
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def compute_mean_std(img_dataset):
    
    # Loading the images with parallel CPU cores.
    desired_cores = min(4, multiprocessing.cpu_count() - 1)
    loader = DataLoader(img_dataset, batch_size=16, num_workers=3) # Each image comes in a batch size of 16 segments.

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for imgs, _ in loader:
        batch_samples = imgs.size(0)
        imgs = imgs.view(batch_samples, imgs.size(1), -1) # Performing a flattening of our images.
        mean += imgs.mean(dim=[0, 2])
        std += imgs.std(dim=[0, 2])
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    
    return mean, std # Returning the resulting mean and std of the R, G, and B channels

# Goal: Calculates the means and std for one image over the 16 sub-sections per image
# Inputs: File-path to the folder for the image containing both the RGB images and the labeled .tif files
def calculate_dataset_stats(img_filepath):
    print(img_filepath)
    dataset = datasets.ImageFolder(img_filepath, transform=transforms.ToTensor())
    return compute_mean_std(dataset)

def get_img_transform(image_dir_path):
    img_mean, img_std = calculate_dataset_stats(image_dir_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=img_mean.tolist(), std=img_std.tolist()) # Using our discovered mean and std data from the image.
    ])

    return transform # Return the resulting transform for this image.

# Creating a class for this specific problem that inherits from Dataset so that we can use DataLoader on it in training.
class CropResidueSegDataset(Dataset):
    def __init__(self, root_directory):
        self.root_dir = root_directory
        self.img_mask_pairs = self._load_img_mask_groups()

    def _load_img_mask_groups(self):
        groups = []

        # We want to navigate through each dataset such as ./Limbaugh1-1m20220328/
        for dataset_dir in os.listdir(self.root_dir):
            dataset_dir_path = os.path.join(self.root_dir, dataset_dir)
            print(f"Dataset directory: {dataset_dir_path}")

            # We need to make a new transform for this image as we will perform the transform/normalization based on parameters from each image.
            img_transform = get_img_transform(dataset_dir_path)

            # We will loop through each IMG folder to extract the image filepath and the mask filepath.
            for img_dir in os.listdir(dataset_dir_path):
                image_folder_path = os.path.join(dataset_dir_path, img_dir)
                print(f"Image directory: {image_folder_path}")

                # If the IMG folder exists, we will extract all of the .jpg images and see if we can pair them with their corresponding .tif files.
                image_files = sorted([file for file in os.listdir(image_folder_path) if file.endswith('.jpg')])

                for file in image_files:

                    image_path = os.path.join(image_folder_path, file) # Result should be <root_path>/<dataset_name>/IMG_0629/IMG_0629_part01.jpg for example

                    # Identify corresponding .tif file
                    mask_path = file.replace("_part", "_res_part", 1) # Should now be "IMG_0629_res_part0.jpg" for example
                    mask_path = mask_path.replace(".jpg", ".tif")
                    mask_path = os.path.join(image_folder_path, mask_path)

                    if os.path.exists(mask_path):
                        groups.append((image_path, mask_path, img_transform))
        
        return groups
    
    # Fetch the length of the dataset
    def __len__(self):
        return len(self.img_mask_pairs)
    
    # Get one image and its corresponding mask. DataLoader will call this function.
    def __getitem__(self, index):
        img_path, mask_path, transform = self.img_mask_pairs[index]

        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")

        if not os.path.exists(mask_path):
            print(f"Image file not found: {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # If a transform is defined for the dataset.
        if transform:
            image = transform(image)

        mask = transforms.ToTensor()(mask).long() # Convert the mask to a Tensor [0, 1]; Squeezes the 4D tensor into a 3D tensor (removes the 1 channel dimension)
        # print(f"Image Tensor: {image}\tMask Tensor: {mask}")

        return image, mask # Return the Tensors


