from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

app = FastAPI(title="MNIST Digit Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

class PredictionInput(BaseModel):
    data: List[float]

@app.on_event("startup")
async def startup_event():
    global model
    model_path = "model.pkl"

    if not os.path.exists(model_path):
        raise RuntimeError(
            f"{model_path} not found. Please train the model locally by running "
            "'python model_train.py' and commit model.pkl to your repository."
        )

    print(f"Loading {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

@app.get("/")
async def root():
    return {"message": "MNIST Digit Classifier API is running"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
async def predict(input_data: PredictionInput):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = np.array(input_data.data).reshape(1, -1)

        if np.max(X) > 1.0:
            X = X / 255.0

        prediction = model.predict(X)[0]
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
    port = int(os.environ.get("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)
