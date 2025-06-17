import json
import base64
from mangum import Mangum

def create_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    app = FastAPI(title="ChestMNIST Inference API")
    
    # Pydantic model for base64 input
    class ImageRequest(BaseModel):
        image: str  # base64 encoded image
        
    class ImageResponse(BaseModel):
        predicted_class: int
        confidence: float
    
    @app.get("/")
    def root():
        return {"message": "ChestMNIST Inference Service is up."}
    
    @app.get("/health")
    def health():
        import os
        possible_paths = [
            "./opt/model",
            "/var/task/model", 
            "./model",
            "model"
        ]
        
        found_path = None
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        return {
            "status": "healthy", 
            "model_path": found_path,
            "current_dir": os.getcwd(),
            "checked_paths": possible_paths
        }
    
    @app.post("/predict", response_model=ImageResponse)
    async def predict_base64(request: ImageRequest):
        """
        Accepts base64 encoded image, returns prediction
        """
        return await _predict_from_base64(request.image)
    
    async def _predict_from_base64(image_b64: str):
        """
        Shared prediction logic for base64 images
        """
        # Import heavy dependencies only when needed
        import os
        import torch
        import mlflow.pyfunc
        from PIL import Image
        from io import BytesIO
        from torchvision import transforms
        import numpy as np
        
        # Force CPU usage for PyTorch
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Model is copied to /var/task/opt/model by Dockerfile
        MODEL_PATH = os.getenv("MODEL_PATH", "./opt/model")
        
        try:
            # Decode base64 image
            try:
                img_data = base64.b64decode(image_b64)
                img = Image.open(BytesIO(img_data)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            x = transform(img).unsqueeze(0)
            
            # Load model (cached after first load)
            if not hasattr(_predict_from_base64, '_model'):
                if not os.path.exists(MODEL_PATH):
                    raise RuntimeError(f"Model directory not found at {MODEL_PATH}")
                
                # Try to load MLflow model first
                if os.path.exists(os.path.join(MODEL_PATH, "MLmodel")):
                    try:
                        # Set environment to force CPU loading
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")  # Suppress MLflow warnings
                            _predict_from_base64._model = mlflow.pyfunc.load_model(MODEL_PATH)
                    except Exception as e:
                        print(f"MLflow loading failed: {e}")
                        # Fallback to direct PyTorch loading
                        _predict_from_base64._model = None
                
                # If MLflow loading failed or no MLmodel file, try direct PyTorch loading
                if not hasattr(_predict_from_base64, '_model') or _predict_from_base64._model is None:
                    # Look for PyTorch model files
                    model_files = []
                    for root, dirs, files in os.walk(MODEL_PATH):
                        for file in files:
                            if file.endswith(('.pth', '.pt')):
                                model_files.append(os.path.join(root, file))
                    
                    if model_files:
                        print(f"Loading PyTorch model from: {model_files[0]}")
                        _predict_from_base64._model = torch.load(
                            model_files[0], 
                            map_location=torch.device('cpu')  # Explicit CPU device
                        )
                    else:
                        raise RuntimeError(f"No model files found in {MODEL_PATH}")
            
            # Predict
            if hasattr(_predict_from_base64._model, 'predict'):
                # MLflow model
                preds = _predict_from_base64._model.predict(x.numpy())
            else:
                # Direct PyTorch model
                _predict_from_base64._model.eval()
                with torch.no_grad():
                    preds = _predict_from_base64._model(x)
                    if hasattr(preds, 'numpy'):
                        preds = preds.numpy()
                    else:
                        preds = preds.detach().cpu().numpy()
            
            # Handle prediction shape
            if len(preds.shape) > 1:
                class_idx = int(np.argmax(preds, axis=1)[0])
                confidence = float(np.max(preds, axis=1)[0])
            else:
                class_idx = int(np.argmax(preds))
                confidence = float(np.max(preds))
            
            return {"predicted_class": class_idx, "confidence": confidence}
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    return app

# Create app instance
app = create_app()

# Lambda handler
handler = Mangum(app)