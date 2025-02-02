# loads in images from google drive and then use utils crop dataset class and pass in
# images
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt

from utils import CropResidueSegDataset

if __name__ == "__main__":
    test1 = CropResidueSegDataset("../test", False)
    dataset_loader = DataLoader(test1, batch_size=16, shuffle=True, num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256,2, kernel_size=(1,1))

    checkpoint = torch.load("./agaid2025_final.pth", map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()
        
    for images, _ in dataset_loader:
        images = images.to(device)

        with torch.no_grad():
            outputs = model(images)["out"] # Get the output Tensor!
            predicted_mask = torch.argmax(outputs, dim=1)

