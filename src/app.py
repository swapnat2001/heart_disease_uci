from fastapi import FastAPI, Response
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from prometheus_client import Counter, generate_latest

logging.basicConfig(filename="api_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")
REQUEST_COUNT = Counter("api_requests_total", "Total Inference Requests")

try:
    model = joblib.load("heart_disease_pipeline.joblib")
except Exception as e:
    model = None
    logging.error(f"Failed to find or parse pipeline matrix binary asset on execution path: {e}")

app = FastAPI(title="Heart Disease MLOps Production API")

class PatientFeatures(BaseModel):
    age: float; sex: float; cp: float; trestbps: float; chol: float
    fbs: float; restecg: float; thalach: float; exang: float
    oldpeak: float; slope: float; ca: float; thal: float

@app.post("/predict")
def predict(data: PatientFeatures):
    REQUEST_COUNT.inc()
    logging.info(f"Received verification transaction context payload: {data}")
    
    if model is None:
        return {"error": "Pipeline binary weights asset uninitialized or corrupted."}
        
    features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, 
                          data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])
    
    prediction = int(model.predict(features)[0])
    confidence = float(max(model.predict_proba(features)[0]))
    
    logging.info(f"Inference resolved successfully. Class: {prediction}, Proba: {confidence}")
    return {"prediction": prediction, "confidence": confidence}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
