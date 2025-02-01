import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from dataset import CropResidueDataset

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths
train_image_dir = "../data/train/images"
train_mask_dir = "../data/train/masks"
val_image_dir = "../data/val/images"
val_mask_dir = "../data/val/masks"

# Data Augmentation
transform = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Load dataset
train_dataset = CropResidueDataset(train_image_dir, train_mask_dir, transform)
val_dataset = CropResidueDataset(val_image_dir, val_mask_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# Load DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1))  # 2 classes: soil vs crop residue
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training function
def train_model(model, train_loader, val_loader, epochs=10):
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"../outputs/checkpoints/deeplab_epoch{epoch+1}.pth")

# Train the model
train_model(model, train_loader, val_loader, epochs=10)
