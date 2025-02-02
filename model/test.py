# loads in images from google drive and then use utils crop dataset class and pass in
# images
import torch 
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import matplotlib.pyplot as plt

from utils import CropResidueSegDataset

if __name__ == "__main__":
    test1 = CropResidueSegDataset("../test", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = deeplabv3_resnet101(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256,2, kernel_size=(1,1))

    model.load_state_dict(torch.load("../agaid2025_final.pth"))
    model.to(device)
    model.eval()

