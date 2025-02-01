import torch
import numpy as np
import multiprocessing
from torchvision import datasets, transforms # We want to use the ToTensor() transform.
from torch.utils.data import DataLoader

def compute_mean_std(img_dataset):
    desired_workers = max(1, multiprocessing.cpu_count() - 1)
    loader = DataLoader(img_dataset, batch_size=16, num_workers=desired_workers)

    mean, std = 0.0, 0.0
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

def calculate_dataset_stats(img_filepath):
    dataset = datasets.ImageFolder(img_filepath, transform=transforms.ToTensor())
    return compute_mean_std(dataset)


