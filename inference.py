import os
import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np
from medmnist import INFO
from mangum import Mangum

# Configuration via environment variables
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")  # No default; must be set in Lambda env
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "chestmnist-learning-demo")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ChestMNIST_ResNet18")
DATA_ROOT = os.getenv("MEDMNIST_DATA_ROOT", "/tmp/medmnist")  # Use /tmp in Lambda if needed
ENABLE_INDEX = os.getenv("ENABLE_INDEX_ENDPOINT", "false").lower() == "true"

app = FastAPI(title="ChestMNIST Inference API")

# Lazy loaded model
_mlflow_model = None

def get_model():
    global _mlflow_model
    if _mlflow_model is None:
        if not TRACKING_URI:
            raise RuntimeError("MLFLOW_TRACKING_URI environment variable is not set")
        # Initialize MLflow settings
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        try:
            _mlflow_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/Production")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow: {e}")
    return _mlflow_model

# Preprocessing pipeline based on MedMNIST stats
info = INFO.get("chestmnist")
if info is None:
    raise RuntimeError("MedMNIST INFO missing for chestmnist")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=tuple(info['mean']), std=tuple(info['std'])),
])

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
    # Preprocess
    x = transform(img).unsqueeze(0).numpy()
    # Load and run model
    try:
        model = get_model()
        preds = model.predict(x)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {"predicted_class": class_idx, "confidence": confidence}

@app.get("/predict/{index}")
def predict_index(index: int):
    """
    Runs inference on a specific test-set index from MedMNIST ChestMNIST dataset.
    Disabled by default in production to avoid timeouts.
    """
    if not ENABLE_INDEX:
        raise HTTPException(status_code=404, detail="Index-based inference disabled in this environment")
    try:
        os.makedirs(DATA_ROOT, exist_ok=True)
        from medmnist import ChestMNIST
        dataset = ChestMNIST(
            split='test',
            root=DATA_ROOT,
            download=True,
            transform=transform,
        )
        img_tensor, label = dataset[index]
    except IndexError:
        raise HTTPException(status_code=404, detail="Index out of range")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {e}")
    x = img_tensor.unsqueeze(0).numpy()
    try:
        model = get_model()
        preds = model.predict(x)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
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
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
