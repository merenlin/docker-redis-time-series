"""
Simple LSTM model trainer for time-series prediction demonstration.
This creates a basic model that can be used with the Flask API.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_data(n_points=1000):
    """Generate sample time-series data for training"""
    # Create timestamps
    start_date = datetime.now() - timedelta(days=n_points)
    timestamps = [start_date + timedelta(days=i) for i in range(n_points)]

    # Generate synthetic time series with trend, seasonality, and noise
    t = np.arange(n_points)
    trend = 0.02 * t  # Linear trend
    seasonal = 10 * np.sin(2 * np.pi * t / 365.25)  # Yearly seasonality
    weekly = 5 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    noise = np.random.normal(0, 2, n_points)  # Random noise

    values = 100 + trend + seasonal + weekly + noise  # Base value of 100

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values
    })

    return df


def create_sequences(data, lookback_window=60):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(lookback_window, len(data)):
        X.append(data[i-lookback_window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """Build LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model


def train_model():
    """Train the LSTM model and save it"""
    logger.info("Starting model training...")

    # Generate sample data
    logger.info("Generating sample data...")
    df = generate_sample_data(1000)

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['value']])

    # Create sequences
    lookback_window = 60
    X, y = create_sequences(scaled_data, lookback_window)

    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    # Build and train model
    model = build_lstm_model((lookback_window, 1))

    logger.info("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    val_loss = model.evaluate(X_val, y_val, verbose=0)

    logger.info(f"Training Loss: {train_loss}")
    logger.info(f"Validation Loss: {val_loss}")

    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': ['value'],
        'lookback_window': lookback_window,
        'training_history': {
            'train_loss': float(train_loss[0]),
            'val_loss': float(val_loss[0]),
            'final_epoch': len(history.history['loss'])
        },
        'created_at': datetime.now().isoformat()
    }

    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the model
    model_path = 'models/lstm_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"Model saved to {model_path}")

    # Also save sample data for testing
    sample_data_path = 'models/sample_data.csv'
    df.to_csv(sample_data_path, index=False)
    logger.info(f"Sample data saved to {sample_data_path}")

    return model_data


if __name__ == "__main__":
    train_model()
