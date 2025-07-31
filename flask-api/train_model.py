"""
Simple time-series model trainer using scikit-learn.
Much lighter than TensorFlow for demonstration purposes.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


def create_features(data, lookback_window=5):
    """Create features for time-series prediction using lagged values"""
    X, y = [], []
    
    for i in range(lookback_window, len(data)):
        # Use past 'lookback_window' values as features
        features = data[i-lookback_window:i]
        target = data[i]
        
        X.append(features)
        y.append(target)
    
    return np.array(X), np.array(y)


def train_model():
    """Train a simple RandomForest model for time-series prediction"""
    logger.info("Generating sample data...")
    df = generate_sample_data(1000)
    
    # Extract values
    values = df['value'].values
    
    # Create features and targets
    logger.info("Creating features...")
    X, y = create_features(values, lookback_window=5)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training RandomForest model...")
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    logger.info(f"Training R² score: {train_score:.4f}")
    logger.info(f"Testing R² score: {test_score:.4f}")
    
    # Save model and scaler
    model_dir = '/app/models' if os.path.exists('/app') else './models'
    os.makedirs(model_dir, exist_ok=True)
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'lookback_window': 5,
        'train_score': train_score,
        'test_score': test_score
    }
    
    model_path = os.path.join(model_dir, 'timeseries_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    train_model()
