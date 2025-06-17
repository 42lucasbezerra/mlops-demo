import os
import argparse
import mlflow
import torch
import torch.nn as nn
from datetime import datetime
from medmnist import INFO, PathMNIST
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from mlflow.models.signature import infer_signature

print('code started')

# Parse command-line arguments
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser(description="Train a ResNet18 on PathMNIST and log to MLflow.")
parser.add_argument(
    "--num_epochs", type=int, default=1,
    help="Number of epochs to train (default: 1)"
)
parser.add_argument(
    "--learning_rate", type=float, default=1e-3,
    help="Learning rate for optimizer (default: 1e-3)"
)
args = parser.parse_args()

# Configure MLflow tracking URI (fallback to localhost)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

# Use a new experiment name for PathMNIST
mlflow.set_experiment("pathmnist-transfer-demo")
print('experiment set')


def main(num_epochs, learning_rate):
    # Ensure root directory exists for dataset
    data_root = "data/medmnist"
    os.makedirs(data_root, exist_ok=True)

    # Generate a descriptive run name
    run_name = f"pathmnist_resnet18_lr{learning_rate}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    aborted = False
    with mlflow.start_run(run_name=run_name) as run:
        # Add organizational tags and hyperparameters
        mlflow.set_tags({
            "dataset": "PathMNIST",
            "model": "ResNet18",
            "framework": "PyTorch"
        })
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Data transforms and loader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = PathMNIST(
            root=data_root,
            split="train",
            download=True,
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

        # Build model
        info = INFO["pathmnist"]
        n_classes = len(info['label'])  # 9 classes
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Training loop
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model.train()
        try:
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device).long()
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_loss = epoch_loss / len(loader)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
                mlflow.log_metric("epoch_loss", avg_loss, step=epoch+1)
        except KeyboardInterrupt:
            print("Training interrupted by user; marking run accordingly and exiting gracefully.")
            mlflow.set_tag("run_status", "interrupted")
            aborted = True

        # Log final metrics and model if not aborted
        if not aborted:
            mlflow.log_metric("final_loss", avg_loss)
            # Log model with signature and input example
            sample_input = next(iter(loader))[0][:1].to(device)
            model.eval()
            with torch.no_grad():
                sample_output = model(sample_input).cpu().numpy()
            sample_input_np = sample_input.cpu().numpy()
            signature = infer_signature(sample_input_np, sample_output)
            mlflow.pytorch.log_model(
                model,
                name="resnet18_pathmnist",
                signature=signature,
                input_example=sample_input_np
            )
    print('run ended' if not aborted else 'run aborted')

if __name__ == "__main__":
    main(args.num_epochs, args.learning_rate)