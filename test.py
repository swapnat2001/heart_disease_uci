import os

# Define the root-level folder directory matrices
os.makedirs("src", exist_ok=True)
os.makedirs("tests", exist_ok=True)
os.makedirs(".github/workflows", exist_ok=True)

# 1. WRITE DEPENDENCIES OVERVIEW (requirements.txt)
with open("requirements.txt", "w") as f:
    f.write("""ucimlrepo
pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow
joblib
fastapi
uvicorn
pydantic
prometheus-client
pytest
flake8
""")

# 2. WRITE TRAINING AUTOMATION PIPELINE (src/train.py)
with open("src/train.py", "w") as f:
    f.write("""import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def run_training():
    mlflow.set_experiment("Heart_Disease_Classification")
    
    with mlflow.start_run(run_name="Production_Random_Forest"):
        # Fetch data safely
        heart_disease = fetch_ucirepo(id=45)
        X = heart_disease.data.features
        y = heart_disease.data.targets
        df = pd.concat([X, y], axis=1)
        
        # Preprocessing and target formatting
        df['num'] = (df['num'] > 0).astype(int)
        df['ca'] = df['ca'].fillna(df['ca'].mode()[0])
        df['thal'] = df['thal'].fillna(df['thal'].mode()[0])
        
        X_clean = df.drop('num', axis=1)
        y_clean = df['num']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.20, random_state=42, stratify=y_clean
        )
        
        # Metrics & parameter logging via telemetry
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("random_state", 42)
        
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate metric vectors
        acc = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", acc)
        
        # Standard target serialization dump
        joblib.dump(model, "heart_disease_pipeline.joblib")
        mlflow.sklearn.log_model(model, "model")
        print(f"Model trained successfully. Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_training()
""")

# 3. WRITE INSTRUMENTED Fast-API BACKEND WEB SERVICE (src/app.py)
with open("src/app.py", "w") as f:
    f.write("""from fastapi import FastAPI, Response
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
""")

# 4. WRITE AUTOMATED TEST SUITE ENGINE (tests/test_core.py)
with open("tests/test_core.py", "w") as f:
    f.write("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def test_data_integrity_and_leakage_constraints():
    # Construct a synthetic data structure mirroring real clinical payload signatures
    mock_data = {
        "age": [63, 67, 67, 37, 41, 56],
        "sex": [1, 1, 1, 1, 0, 1],
        "cp": [1, 4, 4, 3, 2, 2],
        "trestbps": [145, 160, 120, 130, 130, 120],
        "chol": [233, 286, 229, 250, 204, 236],
        "fbs": [1, 0, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 0, 2, 1],
        "thalach": [150, 108, 129, 187, 172, 178],
        "exang": [0, 1, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 3.5, 1.4, 0.8],
        "slope": [3, 2, 2, 3, 1, 1],
        "ca": [0, 3, 2, 0, 0, 0],
        "thal": [6, 3, 7, 3, 3, 3],
        "num": [0, 1, 1, 0, 0, 0]
    }
    df = pd.DataFrame(mock_data)
    X = df.drop("num", axis=1)
    y = df["num"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Assert structural integrity constraints
    assert len(X_train) == 4
    assert len(X_test) == 2
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
""")

# 5. WRITE DOCKER CONTAINER SCHEMA LAYER (Dockerfile)
with open("Dockerfile", "w") as f:
    f.write("""FROM python:3.11-slim
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python src/train.py
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
""")

# 6. WRITE AUTOMATED GITHUB ACTIONS WORKFLOW ( .github/workflows/ci.yml )
with open(".github/workflows/ci.yml", "w") as f:
    f.write("""name: MLOps Automation Engine

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  continuous-integration:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Source Code
      uses: actions/checkout@v4

    - name: Initialize Environment Runtime
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'
        cache: 'pip'

    - name: Install Architecture Core Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Code Quality Auditing (Static Syntax Linting)
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings.
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

    - name: Execute Automated Test Matrix Suite
      run: |
        pytest -v

    - name: Verify Training Pipeline Pipeline & Artifact Generation
      run: |
        python src/train.py

    - name: Archive Production Weights Binary Artifact
      uses: actions/upload-artifact@v4
      with:
        name: heart-disease-pipeline-weights
        path: heart_disease_pipeline.joblib
        retention-days: 7
""")

print("⚙️ Monolithic file successfully parsed and fragmented into clean MLOps Repository layout architecture structure.")