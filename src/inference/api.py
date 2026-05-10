from fastapi import FastAPI

app = FastAPI(title="Llama-3 Domain API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: dict):
    pass
