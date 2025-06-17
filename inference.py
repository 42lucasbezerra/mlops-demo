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
        MODEL_PATH = "/opt/model"
        return {"status": "healthy", "model_exists": os.path.exists(MODEL_PATH)}
    
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
        
        MODEL_PATH = os.getenv("MODEL_PATH", "/opt/model")
        
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
                        raise RuntimeError("No model found")
            
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