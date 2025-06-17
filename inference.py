import os
import mlflow
import mlflow.pyfunc
import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np
from medmnist import INFO
from mangum import Mangum

# Configure MLflow tracking URI (fallback to localhost)
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("chestmnist-learning-demo")

# Model registry name
model_name = "ChestMNIST_ResNet18"
# Load the model from MLflow Model Registry (Production stage)
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# Preprocessing pipeline
info = INFO["chestmnist"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=tuple(info['mean']), std=tuple(info['std'])),
])

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    x = transform(img).unsqueeze(0).numpy()
    preds = model.predict(x)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    return {"predicted_class": class_idx, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

handler = Mangum(app)