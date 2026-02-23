import numpy as np
import torch
from sklearn.model_selection import train_test_split

from dataset import BuildingDataset
from train import train_model


def load_dummy_data():
    # Replace with real loading logic
    X = np.load("data/images.npy")
    y = np.load("data/masks.npy")
    return X, y


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = load_dummy_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = BuildingDataset(X_train, y_train)
    val_dataset = BuildingDataset(X_val, y_val)

    model = train_model(train_dataset, val_dataset, device=device)


if __name__ == "__main__":
    main()
