import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import sys
import os

# Menangkap parameter dari MLProject
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

def main():
    print("Memulai Training di GitHub Actions...")
    
    # Load Data
    if not os.path.exists("rice_processed.csv"):
        print("Error: Dataset rice_processed.csv tidak ditemukan!")
        return

    df = pd.read_csv("rice_processed.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow Run
    with mlflow.start_run():
        # Train Model
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc}")
        
        # Log Metrics & Model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        
        # Simpan Model agar bisa jadi Artifact
        mlflow.sklearn.log_model(rf, "model_random_forest")
        
    print("Training Selesai.")

if __name__ == "__main__":
    main()