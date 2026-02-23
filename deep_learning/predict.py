import torch
import matplotlib.pyplot as plt


def visualize_prediction(model, dataset, device="cuda"):

    model.eval()

    img, true_mask = dataset[0]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred = (pred > 0.5).float().squeeze().cpu().numpy()

    img = img.permute(1, 2, 0).numpy()
    true_mask = true_mask.squeeze().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(true_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

