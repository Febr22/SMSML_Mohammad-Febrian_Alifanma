import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub  # Library baru untuk koneksi ke DagsHub
import os

# === KONFIGURASI ===
DATA_FILE = "riceClassification_preprocessing.csv"
# GANTI DENGAN USERNAME DAN NAMA REPO DAGSHUB ANDA
DAGSHUB_REPO_OWNER = "febrianalif22"  # Contoh: username GitHub/DagsHub Anda
DAGSHUB_REPO_NAME = "SMSML_Mohammad-Febrian_Alifanma" # Nama Repo Anda
EXPERIMENT_NAME = "Eksperimen_Rice_Classification_Advanced"
mlflow.set_tracking_uri("http://127.0.0.1:5500")

def main():
    # 1. Cek Data
    if not os.path.exists(DATA_FILE):
        print(f"Error: File '{DATA_FILE}' tidak ditemukan.")
        return

    # 2. Load Data
    print("Memuat data...")
    df = pd.read_csv(DATA_FILE)
    X = df.drop('Class', axis=1)
    y = df['Class']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Setup DagsHub & MLflow
    # Ini akan otomatis mengkonfigurasi MLflow tracking URI ke DagsHub
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Memulai training ke DagsHub...")
    
    with mlflow.start_run():
        # === A. TRAINING & TUNING ===
        rf = RandomForestClassifier(random_state=42)
        params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
        
        grid = GridSearchCV(rf, params, cv=3, verbose=1, n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        print(f"Parameter Terbaik: {best_params}")

        # === B. EVALUASI ===
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        
        print(f"Akurasi: {acc:.4f}")

        # === C. MANUAL LOGGING (Syarat Advanced) ===
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.sklearn.log_model(best_model, "model_rice_classifier")
        
        # --- Artifact 1: Confusion Matrix ---
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # --- Artifact 2: Feature Importance (Syarat minimal 2 artifact tambahan) ---
        feature_importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        sns.barplot(x=feature_importances, y=feature_importances.index)
        plt.title("Feature Importance")
        plt.xlabel("Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        print("Selesai! Cek repository DagsHub Anda > Tab Experiments.")

if __name__ == "__main__":
    main()