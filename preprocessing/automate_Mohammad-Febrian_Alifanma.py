import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Tentukan path file (agar aman saat dijalankan di GitHub Actions)
# Kita asumsikan script dijalankan dari root repository
INPUT_FILE = "riceClassification.csv"
OUTPUT_FILE = "riceClassification_preprocessing.csv"

def load_data(path):
    if not os.path.exists(path):
        print(f"Error: File {path} tidak ditemukan di lokasi saat ini: {os.getcwd()}")
        return None
    data = pd.read_csv(path)
    print(f"Data berhasil dimuat: {path}")
    return data

def preprocess_data(df):
    # 1. Drop ID
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # 2. Fill NA
    df = df.fillna(df.mean())
    
    # 3. Scaling
    if 'Class' in df.columns:
        target = df['Class']
        features = df.drop(columns=['Class'])
    else:
        target = None
        features = df
    
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    
    if target is not None:
        data_processed = pd.concat([features_scaled, target], axis=1)
    else:
        data_processed = features_scaled
        
    return data_processed

if __name__ == "__main__":
    print("=== Memulai Preprocessing Otomatis ===")
    df = load_data(INPUT_FILE)
    
    if df is not None:
        df_clean = preprocess_data(df)
        df_clean.to_csv(OUTPUT_FILE, index=False)
        print(f"=== Sukses! Data disimpan ke {OUTPUT_FILE} ===")