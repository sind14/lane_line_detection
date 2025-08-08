import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import UNet
from src.information import train_dataset, val_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)


def train(model, train_loader, val_loader, epochs=10):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for images, masks in progress_bar:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                outputs = nn.functional.interpolate(outputs, size=masks.shape[2:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lane_line_model_v_3.pth")
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        if epoch > 2 and val_loss > best_val_loss:
            print("Reducing learning rate.")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5


if __name__ == "__main__":
    torch.cuda.empty_cache()
    train(model, train_loader, val_loader, epochs=10)
