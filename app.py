from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import LlamaInferrer
import uvicorn
import os

app = FastAPI(title="Llama-3 Fraud Detection API")

# Initialize the Inferrer (Loads model once on startup)
# Note: Ensure model adapters are present in path
try:
    inferrer = LlamaInferrer()
except Exception as e:
    print(f"Model initialization failed: {e}")
    inferrer = None

class TransactionData(BaseModel):
    details: str

@app.get("/")
def read_root():
    return {"message": "Llama-3 Fraud Detection API is Live!"}

@app.post("/predict")
def predict_fraud(data: TransactionData):
    if not inferrer:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        prediction = inferrer.predict_fraud(data.details)
        return {
            "transaction": data.details,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
