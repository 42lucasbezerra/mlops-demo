import os
import mlflow
import torch
import torch.nn as nn
from medmnist import INFO, ChestMNIST
from torchvision import transforms, models
from torch.utils.data import DataLoader, Subset

print('code started')

# Configure MLflow tracking URI (fallback to localhost)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("chestmnist-transfer-demo")

print('experiment set')


def main():
    print('starting run')
    with mlflow.start_run():
        # Load ChestMNIST dataset
        info = INFO["chestmnist"]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(info['mean']), std=tuple(info['std'])),
        ])
        dataset = ChestMNIST(
            root="data/medmnist",
            split="train",
            download=True,
            transform=transform
        )
        # Subsample for quick CPU runs
        #subset = Subset(dataset, list(range(len(dataset) // 5)))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Build model (freeze backbone, fine-tune head)
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, info['n_classes'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train for n epochs
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        model.train()
        num_epochs = 1
        for n in range(num_epochs):
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device).long()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # 4) Log parameters, metrics, and model to MLflow
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("dataset", "ChestMNIST")
        mlflow.log_metric("final_loss", loss.item())
        mlflow.pytorch.log_model(model, "resnet18_chestmnist")

if __name__ == "__main__":
    main()