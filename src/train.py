import pandas as pd
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
