from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import logging
import io
import matplotlib
# Force matplotlib to use a non-interactive backend suited for server applications
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from prometheus_client import CollectorRegistry, Counter, generate_latest

logging.basicConfig(filename="api_logs.log", level=logging.INFO, format="%(asctime)s - %(message)s")
PROM_REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter("api_requests_total", "Total Inference Requests", registry=PROM_REGISTRY)

try:
    model = joblib.load("heart_disease_pipeline.joblib")
except Exception as e:
    model = None
    logging.error(f"Failed to find or parse pipeline matrix binary asset on execution path: {e}")

app = FastAPI(title="Heart Disease MLOps Production API")

# Reusable function to fetch and clean raw data context safely
def load_and_clean_dataset():
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        # Apply the exact preprocessing cleaning pipeline used in training
        df['num'] = (df['num'] > 0).astype(int)
        df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
        df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
        return df
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {str(e)}")

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
    return Response(generate_latest(PROM_REGISTRY), media_type="text/plain")

@app.get("/eda-summary")
def get_eda_summary():
    """Generates structural metadata shape dimensions and sample entries for reports."""
    df = load_and_clean_dataset()
    return {
        "dataset_shape": {
            "rows": df.shape[0],
            "columns": df.shape[1]
        },
        "column_names": list(df.columns),
        "sample_entries": df.head(5).to_dict(orient="records")
    }

@app.get("/eda-plots")
def get_eda_plots():
    """Generates a comprehensive grid of histograms and a correlation matrix heatmap."""
    df = load_and_clean_dataset()
    
    # Create a unified figure layout for data verification screenshots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Age Distribution Histogram
    sns.histplot(df['age'], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Age Feature Distribution')
    axes[0].set_xlabel('Age')
    
    # 2. Cholesterol Distribution Histogram
    sns.histplot(df['chol'], kde=True, ax=axes[1], color='salmon')
    axes[1].set_title('Cholesterol Feature Distribution')
    axes[1].set_xlabel('Cholesterol (mg/dl)')
    
    # 3. Correlation Matrix Heatmap
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[2], cbar=True)
    axes[2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    
    # Save chart objects directly to memory without locking local storage files
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    plt.close(fig)
    img_buf.seek(0)
    
    return Response(content=img_buf.getvalue(), media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)