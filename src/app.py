from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.inference import LlamaInferrer
from src.utils import setup_logging

# Initialize Logging
setup_logging()

app = FastAPI(title="Llama-3 Domain Fine-Tuned API", version="1.0.0")

# Lazy initialize inferrer
inferrer = None

class QueryRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class QueryResponse(BaseModel):
    response: str

@app.on_event("startup")
def startup_event():
    global inferrer
    # Inferrer initialization can be slow, so we handle it here
    # inferrer = LlamaInferrer()
    pass

@app.get("/")
def read_root():
    return {"status": "Model API is up and running"}

@app.post("/generate", response_model=QueryResponse)
def generate_text(request: QueryRequest):
    """
    Endpoint to generate text using the fine-tuned Llama-3 model.
    """
    try:
        # response = inferrer.generate(request.prompt, request.max_tokens)
        response = "API is working (Inference placeholder)"
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
