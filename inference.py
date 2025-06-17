import os
import json
import pickle
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
from io import BytesIO
from torchvision import transforms
import numpy as np
from medmnist import INFO
from mangum import Mangum

# Configuration via environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "/opt/model")  # Path where model is stored locally
DATA_ROOT = os.getenv("MEDMNIST_DATA_ROOT", "/tmp/medmnist")
ENABLE_INDEX = os.getenv("ENABLE_INDEX_ENDPOINT", "false").lower() == "true"

app = FastAPI(title="ChestMNIST Inference API")

# Lazy loaded model
_model = None
_model_metadata = None

def load_local_model():
    """Load model from local filesystem (downloaded by deployment script)"""
    global _model, _model_metadata
    
    if _model is None:
        try:
            # Check if model path exists
            if not os.path.exists(MODEL_PATH):
                raise RuntimeError(f"Model path {MODEL_PATH} does not exist")
            
            # Look for MLmodel file to understand the model format
            mlmodel_path = os.path.join(MODEL_PATH, "MLmodel")
            if os.path.exists(mlmodel_path):
                # Load as MLflow model
                import mlflow.pyfunc
                _model = mlflow.pyfunc.load_model(MODEL_PATH)
            else:
                # Try to load as PyTorch model directly
                model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(('.pth', '.pt', '.pkl'))]
                if not model_files:
                    raise RuntimeError(f"No model files found in {MODEL_PATH}")
                
                model_file = os.path.join(MODEL_PATH, model_files[0])
                if model_file.endswith('.pkl'):
                    with open(model_file, 'rb') as f:
                        _model = pickle.load(f)
                else:
                    _model = torch.load(model_file, map_location='cpu')
                    
            # Try to load metadata if available
            metadata_path = os.path.join(MODEL_PATH, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    _model_metadata = json.load(f)
                    
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")
    
    return _model

def get_model():
    """Get the loaded model"""
    return load_local_model()

# Preprocessing pipeline based on MedMNIST stats
try:
    info = INFO.get("chestmnist")
    if info is None:
        # Fallback values if INFO is not available
        info = {
            'mean': [0.485, 0.456, 0.406],  # ImageNet defaults
            'std': [0.229, 0.224, 0.225]
        }
except Exception:
    # Fallback values
    info = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=tuple(info['mean']), std=tuple(info['std'])),
])

@app.get("/")
def root():
    return {"message": "ChestMNIST Inference Service is up."}

@app.get("/health")
def health():
    """Health check endpoint"""
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": model is not None}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image upload, runs the model, returns predicted class and confidence.
    """
    try:
        data = await file.read()
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    
    # Preprocess
    try:
        x = transform(img).unsqueeze(0)
        
        # Convert to numpy if needed for MLflow model
        model = get_model()
        
        # Handle different model types
        if hasattr(model, 'predict'):
            # MLflow pyfunc model
            x_numpy = x.numpy()
            preds = model.predict(x_numpy)
        elif hasattr(model, 'forward') or callable(model):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    x = x.cuda()
                    model = model.cuda()
                preds = model(x)
                if hasattr(preds, 'cpu'):
                    preds = preds.cpu().numpy()
                else:
                    preds = np.array(preds)
        else:
            raise RuntimeError("Unknown model type")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    # Process predictions
    try:
        if len(preds.shape) > 1:
            class_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
        else:
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            
        return {"predicted_class": class_idx, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction processing failed: {str(e)}")

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
    
    try:
        x = img_tensor.unsqueeze(0)
        model = get_model()
        
        # Handle different model types (same as above)
        if hasattr(model, 'predict'):
            x_numpy = x.numpy()
            preds = model.predict(x_numpy)
        elif hasattr(model, 'forward') or callable(model):
            model.eval()
            with torch.no_grad():
                if torch.cuda.is_available():
                    x = x.cuda()
                    model = model.cuda()
                preds = model(x)
                if hasattr(preds, 'cpu'):
                    preds = preds.cpu().numpy()
                else:
                    preds = np.array(preds)
        else:
            raise RuntimeError("Unknown model type")
            
        if len(preds.shape) > 1:
            class_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(np.max(preds, axis=1)[0])
        else:
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
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