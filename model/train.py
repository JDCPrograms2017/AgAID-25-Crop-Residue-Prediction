import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet101
from utils import CropResidueSegDataset
import multiprocessing

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_data_path = "../images_512/label/residue_background"

dataset = CropResidueSegDataset(img_data_path)

desired_workers = max(1, multiprocessing.cpu_count() - 1)
dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Load DeepLabV3 model
model = deeplabv3_resnet101(pretrained=True)
model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1))  # 2 classes: soil vs crop residue
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# Training function
def train_model(model, dataloader, epochs=10):
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

        # Save model checkpoint
        torch.save(model.state_dict(), f"../outputs/checkpoints/deeplab_epoch{epoch+1}.pth")

# Train the model
train_model(model, dataset_loader, epochs=10)
