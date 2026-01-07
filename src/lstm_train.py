import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def build_baseline_lstm(input_shape):
    model = models.Sequential([
        # Modern way to define input shape to avoid the UserWarning
        layers.Input(shape=input_shape), 
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm(file_path):
    df = pd.read_csv(file_path)
    
    # Remove the target and non-numeric metadata to prevent data leakage
    drop_cols = ['target', 'tourney_date', 'p1_name', 'p2_name']
    X_raw = df.drop(columns=drop_cols, errors='ignore')
    y = df['target'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # Reshape to (samples, 1, features) for baseline
    X = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1])) 

    # Train
    input_shape = (X.shape[1], X.shape[2]) 
    model = build_baseline_lstm(input_shape)
    
    print(f"Training on {X.shape[2]} features...")
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)
    
    # Save in modern .keras format
    save_path = Path(file_path).parent.parent.parent / "models" / "baseline_lstm.keras"
    save_path.parent.mkdir(exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent
    test_path = base_path / "data" / "processed" / "atp_with_elo.csv"
    if test_path.exists():
        train_lstm(test_path)