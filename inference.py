import os
import mlflow
import mlflow.pyfunc
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np
from medmnist import INFO, ChestMNIST
from mangum import Mangum

# Configuration via environment variables
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "chestmnist-learning-demo")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ChestMNIST_ResNet18")
DATA_ROOT = os.getenv("MEDMNIST_DATA_ROOT", "./data")

# Initialize MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Load the model once at cold start
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")

# Preprocessing pipeline based on MedMNIST stats
info = INFO["chestmnist"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=tuple(info['mean']), std=tuple(info['std'])),
])

app = FastAPI(title="ChestMNIST Inference API")

@app.get("/")
def root():
    return {"message": "ChestMNIST Inference Service is up."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs the model, returns predicted class and confidence.
    """
    try:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Preprocess and predict
    x = transform(img).unsqueeze(0).numpy()
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {"predicted_class": class_idx, "confidence": confidence}

@app.get("/predict/{index}")
def predict_index(index: int):
    """
    Runs inference on a specific test-set index from MedMNIST ChestMNIST dataset.
    """
    try:
        dataset = ChestMNIST(
            split='test',
            root=DATA_ROOT,
            download=True,
            transform=transform,
        )
        img_tensor, label = dataset[index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Index out of range")
    except Exception:
        raise HTTPException(status_code=500, detail="Error loading dataset")

    x = img_tensor.unsqueeze(0).numpy()
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {
        "index": index,
        "actual_label": int(label),
        "predicted_class": class_idx,
        "confidence": confidence,
    }

# AWS Lambda handler via Mangum
handler = Mangum(app)

if __name__ == "__main__":
    # For local debugging
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
