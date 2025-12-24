import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# 1. Load Data
# Pastikan path ini sesuai dengan lokasi file CSV Anda di folder submission
df = pd.read_csv('riceClassification_preprocessing.csv')

# Pisahkan Fitur (X) dan Target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# PERBAIKAN UTAMA: Menggunakan mlflow.start_run() agar artifact tersimpan
# ==============================================================================
mlflow.set_experiment("Rice_Classification_Experiment")

with mlflow.start_run():
    # Definisi Hyperparameter
    n_estimators = 100
    random_state = 42
    
    # Log Parameter (Mencatat settingan model)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    
    # 3. Train Model
    print("Sedang melatih model...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Akurasi Model: {accuracy}")
    
    # Log Metric (Mencatat hasil akurasi)
    mlflow.log_metric("accuracy", accuracy)
    
    # 5. Simpan Model sebagai Artifact (PENTING!)
    print("Menyimpan model ke MLflow...")
    mlflow.sklearn.log_model(model, "model_random_forest")
    
    print("Selesai! Artifact tersimpan.")