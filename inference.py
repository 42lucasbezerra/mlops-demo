import json
from mangum import Mangum

# Create FastAPI app with minimal imports at module level
def create_app():
    from fastapi import FastAPI, HTTPException, File, UploadFile
    app = FastAPI(title="ChestMNIST Inference API")
    
    @app.get("/")
    def root():
        return {"message": "ChestMNIST Inference Service is up."}
    
    @app.get("/health")
    def health():
        import os
        MODEL_PATH = "/opt/model"
        return {"status": "healthy", "model_exists": os.path.exists(MODEL_PATH)}
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        # Import heavy dependencies only when needed
        import os
        import torch
        import mlflow.pyfunc
        from PIL import Image
        from io import BytesIO
        from torchvision import transforms
        import numpy as np
        
        MODEL_PATH = "/opt/model"
        
        try:
            # Read image
            data = await file.read()
            img = Image.open(BytesIO(data)).convert("RGB")
            
            # Quick transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            x = transform(img).unsqueeze(0)
            
            # Load model (cached after first load)
            if not hasattr(predict, '_model'):
                if os.path.exists(os.path.join(MODEL_PATH, "MLmodel")):
                    predict._model = mlflow.pyfunc.load_model(MODEL_PATH)
                else:
                    model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith(('.pth', '.pt'))]
                    if model_files:
                        predict._model = torch.load(os.path.join(MODEL_PATH, model_files[0]), map_location='cpu')
                    else:
                        raise RuntimeError("No model found")
            
            # Predict
            if hasattr(predict._model, 'predict'):
                preds = predict._model.predict(x.numpy())
            else:
                predict._model.eval()
                with torch.no_grad():
                    preds = predict._model(x).numpy()
            
            class_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))
            
            return {"predicted_class": class_idx, "confidence": confidence}
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

# Create app instance (this happens during init)
app = create_app()

# Lambda handler
handler = Mangum(app)