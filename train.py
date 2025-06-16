import os
import argparse
import mlflow
import torch
import torch.nn as nn
from medmnist import INFO, ChestMNIST
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature

print('code started')

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a ResNet18 on ChestMNIST and log to MLflow.")
parser.add_argument(
    "--num_epochs", type=int, default=1,
    help="Number of epochs to train (default: 1)"
)
args = parser.parse_args()

# Configure MLflow tracking URI (fallback to localhost)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("chestmnist-transfer-demo")

print('experiment set')

def main(num_epochs):
    # Ensure root directory exists for dataset
    data_root = "data/medmnist"
    os.makedirs(data_root, exist_ok=True)

    with mlflow.start_run():
        # Load ChestMNIST dataset and apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ChestMNIST(
            root=data_root,
            split="train",
            download=True,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        # Build model (freeze backbone, fine-tune head for multi-label)
        info = INFO["chestmnist"]
        n_classes = len(info['label'])  # 14 labels
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Train for specified epochs with BCEWithLogitsLoss
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device).float()
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(loader):.4f}")

        # Log parameters, metrics
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("dataset", "ChestMNIST")
        mlflow.log_metric("final_loss", loss.item())

        # Infer model signature and log model with input example
        sample_batch = next(iter(loader))[0][:1].to(device)
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_batch).cpu().numpy()
        sample_input = sample_batch.cpu().numpy()
        signature = infer_signature(sample_input, sample_output)
        mlflow.pytorch.log_model(
            model,
            "resnet18_chestmnist",
            signature=signature,
            input_example=sample_input
        )

if __name__ == "__main__":
    main(args.num_epochs)
