# loads in images from google drive and then use utils crop dataset class and pass in
# images
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt
import tifffile as tiff
import os

from utils import CropResidueSegDataset

if __name__ == "__main__":
    test1 = CropResidueSegDataset("/scratch/project/hackathon/data/CropResiduePredictionChallenge/test/", False)
    dataset_loader = DataLoader(test1, batch_size=16, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256,2, kernel_size=(1,1))

    checkpoint = torch.load("/home/joshua.chadwick/outputs/final/agaid2025_final.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
        
    for batch_index, images in enumerate(dataset_loader):
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)["out"] # Get the output Tensor!
            predicted_mask = torch.argmax(outputs, dim=1).squeeze().cpu().numpy()

        for i in range(predicted_mask.shape[0]):
            outfile_name = f"IMG_{i + 1}_segmented.tif"
            outfile_path = os.path.join("../outputs/seg_imgs", outfile_name)
            
            tiff.imwrite(outfile_path, predicted_mask[i]) # Output the segmented mask as a .tif

