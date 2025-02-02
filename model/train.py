import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models.segmentation import deeplabv3_resnet101
from utils import CropResidueSegDataset
import multiprocessing

# Training function
def train_model(model, dataloader, epochs=10):

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print("Training the DeepLabV3+ Model\n-----------------------------")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            print("Images and masks have been properly loaded!")
            optimizer.zero_grad()

            outputs = model(images)["out"] # Model freezes here in training mode...

            print(f"Output shape: {outputs.shape}, Mask shape: {masks.shape}")
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"../outputs/checkpoints/deeplab_epoch{epoch+1}.pth")


if __name__ == '__main__':
    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")

    img_data_path = "images_512\label\\residue_background"

    dataset = CropResidueSegDataset(img_data_path)
    print(f"Dataset loaded! Size: {len(dataset)}")

    dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Load DeepLabV3 model
    model = deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1,1))  # 2 classes: soil vs crop residue
    model = model.to(device)
    print("Model loaded!")
    
    # Train the model
    train_model(model, dataset_loader, epochs=10)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for images, masks in dataset_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)["out"]
            print(f"Output shape: {outputs.shape}")
            print(f"Mask shape: {masks.shape}")
            break  # Exit after one batch for testing
