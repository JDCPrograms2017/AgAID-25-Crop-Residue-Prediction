import os
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from dataset import CropResidueDataset  # Ensure correct import

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === FIXED: Get the correct absolute path to images_512 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get project root
img_data_path = os.path.join(BASE_DIR, "images_512", "original")  # Correct path for images
mask_data_path = os.path.join(BASE_DIR, "images_512", "label")    # Correct path for masks

# Ensure paths exist before running
if not os.path.exists(img_data_path) or not os.path.exists(mask_data_path):
    raise FileNotFoundError(f"❌ Dataset folders not found! Check paths:\n{img_data_path}\n{mask_data_path}")

# === Load Dataset ===
dataset = CropResidueDataset(img_data_path, mask_data_path)  # Pass both arguments
desired_workers = max(1, multiprocessing.cpu_count() - 1)  # Dynamically assign workers
dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=desired_workers)

# === Load DeepLabV3 Model ===
model = deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1))  # 2 classes: soil vs crop residue
model = model.to(device)

# === Define Loss Function & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# === Training Function ===
def train_model(model, dataloader, epochs=10):
    best_loss = float("inf")
    checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save best model (only if loss improves)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_deeplabv3.pth"))
            print(f"✅ Saved best model (Epoch {epoch+1})")

# === Train the Model ===
train_model(model, dataset_loader, epochs=10)
