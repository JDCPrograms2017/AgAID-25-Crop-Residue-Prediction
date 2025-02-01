import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet101

# Load pre-trained DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.eval()

# Device setup (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# === RUN SEGMENTATION ===
def analyze_crop_residue(tensor_data):
    """
    Takes in a pre-normalized tensor (from Josh's pipeline), 
    runs DeepLabV3 inference, and generates a segmentation mask.
    """
    tensor_data = tensor_data.to(device)  # Move to GPU if available

    with torch.no_grad():
        output = model(tensor_data)["out"]

    # Convert output to segmentation mask
    segmentation_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return segmentation_mask

# === DISPLAY FUNCTION ===
def show_segmentation(image_tensor, segmentation_mask):
    """
    Visualizes the original image (already in tensor format from Josh)
    and its corresponding segmentation mask.
    """
    # Convert tensor to NumPy image (assuming it's in (C, H, W) format)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask, cmap="jet")
    plt.title("Segmentation Mask")

    plt.show()

# === MAIN EXECUTION (Example) ===
if __name__ == "__main__":
    # Assume `tensor_data` is coming directly from Josh’s pipeline
    example_tensor = torch.randn(1, 3, 512, 512)  # Example input tensor (Replace with actual data)

    mask = analyze_crop_residue(example_tensor)
    show_segmentation(example_tensor, mask)
