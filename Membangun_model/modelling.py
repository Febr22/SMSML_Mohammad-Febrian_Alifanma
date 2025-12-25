import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

# 1. Load Data
df = pd.read_csv('riceClassification_preprocessing.csv')
X = df.drop(columns=['Class'])
y = df['Class']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# PERBAIKAN: Menggunakan mlflow.autolog() sesuai permintaan Reviewer
# ==============================================================================
mlflow.set_experiment("Rice_Classification_Autolog")
mlflow.autolog() # Ini akan otomatis melog model, parameter, dan metrik standar

with mlflow.start_run():
    # 3. Train Model
    print("Sedang melatih model dengan Autolog...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi Manual (Tambahan)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc}")

    # 5. Membuat Confusion Matrix (Reviewer minta ini ada gambarnya)
    print("Membuat Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Simpan gambar lalu log sebagai artifact
    plt.savefig("training_confusion_matrix.png")
    mlflow.log_artifact("training_confusion_matrix.png")
    
    print("Selesai! Cek MLflow UI.")