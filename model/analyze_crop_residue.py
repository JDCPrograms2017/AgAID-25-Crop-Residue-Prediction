import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Device setup (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === NORMALIZATION PARAMETERS (Ensure compatibility with Josh's tensor handling) ===
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# === IMAGE PROCESSING FUNCTION ===
def process_image(image_path):
    """Loads and normalizes an image to be compatible with DeepLabV3."""
    image = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Resize((512, 512)),  # Match model input
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),  # Apply Josh's normalization method
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, image

# === RUN SEGMENTATION ===
def analyze_crop_residue(image_path):
    """Runs DeepLabV3 inference on a crop residue image."""
    input_tensor, original_image = process_image(image_path)

    with torch.no_grad():
        output = model(input_tensor)["out"]

    # Convert output to segmentation mask
    segmentation_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Display results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask, cmap="jet")
    plt.title("Segmentation Mask")

    plt.show()

    return segmentation_mask

# === MAIN EXECUTION ===
if __name__ == "__main__":
    IMAGE_PATH = "crop_residue.jpg"  # Change this to your actual image
    mask = analyze_crop_residue(IMAGE_PATH)
