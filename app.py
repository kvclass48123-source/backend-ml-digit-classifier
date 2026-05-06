from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

# Import training function if we need to train on the fly
# Note: In a real production environment, you'd probably do this offline
try:
    from model_train import train_model
except ImportError:
    # If used as a standalone script
    def train_model():
        pass

app = FastAPI(title="MNIST Digit Classifier")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

class PredictionInput(BaseModel):
    # Expecting a flat list of 784 pixels (for 28x28 image)
    data: List[float]

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "model.pkl"
    
    if not os.path.exists(model_path):
        print(f"{model_path} not found. Training model...")
        # Since training might take a while, in a real app you might run this as a background task
        # or have it pre-trained. For this requirement, we train it now.
        from model_train import train_model
        train_model()
    
    print(f"Loading {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

@app.get("/")
async def root():
    return {"message": "MNIST Digit Classifier API is running"}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare data (reshape to 1x784)
        X = np.array(input_data.data).reshape(1, -1)
        
        # Scale if necessary (assuming input is 0-255, we scale to 0-1 as model was trained on scaled data)
        # However, it's safer if the client scales it. Let's check max value.
        if np.max(X) > 1.0:
            X = X / 255.0
            
        # Get prediction
        prediction = model.predict(X)[0]
        
        # Get probabilities for confidence
        probabilities = model.predict_proba(X)[0]
        confidence = float(np.max(probabilities))
        
        return {
            "prediction": str(prediction),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Use port 3000 as required by the platform environment
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
