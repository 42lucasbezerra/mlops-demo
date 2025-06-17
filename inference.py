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
            os.getenv("MODEL_PATH", "/opt/model"),
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
        
        # Check multiple possible model locations
        possible_paths = [
            os.getenv("MODEL_PATH", "/opt/model"),  # Lambda layer location
            "/var/task/model",                      # Lambda function code location
            "./model",                              # Relative path
            "model"                                 # Current directory
        ]
        
        MODEL_PATH = None
        for path in possible_paths:
            if os.path.exists(path):
                MODEL_PATH = path
                break
        
        if MODEL_PATH is None:
            # Debug info
            current_dir = os.getcwd()
            dir_contents = os.listdir(current_dir) if os.path.exists(current_dir) else []
            raise HTTPException(
                status_code=500, 
                detail=f"Model not found. Checked paths: {possible_paths}. Current dir: {current_dir}, Contents: {dir_contents}"
            )
        
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
                if os.path.exists(os.path.join(MODEL_PATH, "MLmodel")):
                    _predict_from_base64._model = mlflow.pyfunc.load_model(MODEL_PATH)
                else:
                    model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(('.pth', '.pt'))]
                    if model_files:
                        _predict_from_base64._model = torch.load(
                            os.path.join(MODEL_PATH, model_files[0]), 
                            map_location='cpu'
                        )
                    else:
                        raise RuntimeError(f"No model files found in {MODEL_PATH}")
            
            # Predict
            if hasattr(_predict_from_base64._model, 'predict'):
                preds = _predict_from_base64._model.predict(x.numpy())
            else:
                _predict_from_base64._model.eval()
                with torch.no_grad():
                    preds = _predict_from_base64._model(x).numpy()
            
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