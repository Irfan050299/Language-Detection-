from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_pipeline

app = FastAPI()


class TextIn(BaseModel):
    text:str

class PredictionOut(BaseModel):
    language: str

@app.get("/")
def home():
    return {"Hello this is language detection app"}

@app.post("/predict",response_model=PredictionOut)
def predict(payload:TextIn):

    try:
            language = predict_pipeline(payload.text)
            return PredictionOut(language=language)
    except Exception as e:  # Catch potential errors from the prediction pipeline
            raise ValueError(f"An error occurred during prediction: {str(e)}")
