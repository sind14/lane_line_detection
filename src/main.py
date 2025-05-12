import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.model import UNet
from src.information import train_dataset, val_dataset, test_dataset

# Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("best_lane_line_model_v_2.pth", map_location=device))
model.eval()

# Завантажувачі
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# Функція візуалізації
def visualize_predictions(model, loader, title=""):
    images, true_masks = next(iter(loader))
    images, true_masks = images.to(device), true_masks.to(device)

    with torch.no_grad():
        outputs = model(images)
        pred_masks = torch.sigmoid(outputs)
        pred_masks = (pred_masks > 0.5).float()

    images = images.cpu()
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()

    num_samples = min(4, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 3 * num_samples))
    fig.suptitle(title, fontsize=16)

    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        ax_img, ax_true, ax_pred = axes[i] if num_samples > 1 else axes[0]

        ax_img.imshow(images[i][0], cmap="gray")
        ax_img.set_title("Image")
        ax_img.axis("off")

        ax_true.imshow(true_masks[i][0], cmap="gray")
        ax_true.set_title("True Mask")
        ax_true.axis("off")

        ax_pred.imshow(pred_masks[i][0], cmap="gray")
        ax_pred.set_title("Predicted Mask")
        ax_pred.axis("off")

    plt.tight_layout()
    plt.show()


# Візуалізація на всіх наборах
visualize_predictions(model, train_loader, title="Train Set Results")
visualize_predictions(model, val_loader, title="Validation Set Results")
visualize_predictions(model, test_loader, title="Test Set Results")
