import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Load Data
if os.path.exists("riceClassification_preprocessing.csv"):
    df = pd.read_csv("riceClassification_preprocessing.csv")
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Eval
    y_pred = model.predict(X_test)
    print(f"Accuracy Basic Model: {accuracy_score(y_test, y_pred)}")
else:
    print("File dataset tidak ditemukan.")