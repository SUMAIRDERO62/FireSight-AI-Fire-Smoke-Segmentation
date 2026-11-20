from fastapi import FastAPI, File, HTTPException, UploadFile

from firesight.inference.predictor import Predictor

app = FastAPI(title="FireSight AI API", version="0.1.0")
predictor = Predictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="An image file is required")
    result = predictor.predict_bytes(await file.read())
    return {
        "label": result.label,
        "confidence": result.confidence,
        "probabilities": result.probabilities,
    }

