"""
Production-ready retrainer for time-series models.
Demonstrates the "External State Store" pattern for model updates.
"""

import numpy as np
import pandas as pd
import pickle
import os
import redis
import json
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/timeseries_model.pkl')
DATA_PATH = os.getenv('DATA_PATH', '/app/data')
LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', 20))

class TimeSeriesRetrainer:
    def __init__(self):
        self.redis_client = None
        self.connect_redis()
        
    def connect_redis(self):
        """Connect to Redis for accessing recent data and storing metadata"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def load_all_data_from_redis(self):
        """Load ALL historical data from Redis (single source of truth)"""
        if not self.redis_client:
            logger.warning("No Redis connection, generating sample data")
            return self.generate_sample_data()
            
        try:
            # Get all active series
            active_series = self.redis_client.smembers("series:active")
            
            if not active_series:
                logger.warning("No active series found in Redis, generating sample data")
                return self.generate_sample_data()
            
            all_data = []
            total_points = 0
            
            for series_id in active_series:
                logger.info(f"Loading data for series: {series_id}")
                
                # Get all timestamps for this series (sorted by time)
                timeline_key = f"timeline:{series_id}"
                timestamps = self.redis_client.zrange(timeline_key, 0, -1)
                
                series_points = 0
                for timestamp in timestamps:
                    point_key = f"timeseries:{series_id}:{timestamp}"
                    point_data = self.redis_client.get(point_key)
                    
                    if point_data:
                        data_point = json.loads(point_data)
                        all_data.append(data_point)
                        series_points += 1
                
                logger.info(f"Loaded {series_points} points for series {series_id}")
                total_points += series_points
            
            if not all_data:
                logger.warning("No data found in Redis, generating sample data")
                return self.generate_sample_data()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {total_points} total data points from Redis across {len(active_series)} series")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from Redis: {e}")
            logger.warning("Falling back to sample data generation")
            return self.generate_sample_data()

    def save_training_data_summary_to_redis(self, df):
        """Save training data summary to Redis for reference"""
        try:
            summary = {
                'total_points': len(df),
                'series_count': df['series_id'].nunique() if 'series_id' in df.columns else 1,
                'date_range': {
                    'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                    'end': df['timestamp'].max().isoformat() if len(df) > 0 else None
                },
                'training_completed_at': datetime.utcnow().isoformat()
            }
            
            self.redis_client.set("training:data_summary", json.dumps(summary))
            logger.info(f"Training data summary saved to Redis: {summary}")
            
        except Exception as e:
            logger.warning(f"Failed to save training summary to Redis: {e}")

    def run_retraining(self):
        """Main retraining workflow - Redis only"""
        logger.info("Starting Redis-only model retraining workflow")
        
        try:
            # 1. Load ALL data from Redis (single source of truth)
            df = self.load_all_data_from_redis()
            
            if len(df) < 50:
                logger.error(f"Insufficient data for training (got {len(df)}, need at least 50 samples)")
                return False
            
            # 2. Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) < 25:
                logger.error(f"Insufficient feature samples for training (got {len(X)}, need at least 25)")
                return False
            
            # 3. Train model
            model, scaler, metrics = self.train_model(X, y)
            
            # 4. Save model and metadata
            self.save_model(model, scaler, metrics)
            
            # 5. Save training data summary to Redis
            self.save_training_data_summary_to_redis(df)
            
            logger.info("Redis-only retraining completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

    def load_recent_data_from_redis(self):
        """Load recent data from Redis cache"""
        recent_data = []
        
        if not self.redis_client:
            return pd.DataFrame()
            
        try:
            # Get all time series keys
            pattern = "timeseries:*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                series_data = self.redis_client.get(key)
                if series_data:
                    series_points = json.loads(series_data)
                    for point in series_points:
                        point['series_id'] = key.split(':')[1]
                        recent_data.append(point)
            
            if recent_data:
                df = pd.DataFrame(recent_data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                logger.info(f"Loaded {len(df)} recent data points from Redis")
                return df.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error loading recent data from Redis: {e}")
            
        return pd.DataFrame()

    def generate_sample_data(self, n_points=2000):
        """Generate synthetic time-series data for demonstration"""
        logger.info("Generating synthetic training data")
        
        start_date = datetime.now() - timedelta(days=n_points)
        timestamps = [start_date + timedelta(hours=i) for i in range(n_points)]
        
        # Generate realistic time series with multiple patterns
        t = np.arange(n_points)
        trend = 0.01 * t
        daily_seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
        weekly_seasonal = 5 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
        noise = np.random.normal(0, 1, n_points)
        
        values = 100 + trend + daily_seasonal + weekly_seasonal + noise
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'series_id': 'synthetic'
        })
        
        return df

    def prepare_features(self, df):
        """Create features for time-series prediction"""
        logger.info("Preparing features for training")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        features = []
        targets = []
        
        # Create sliding window features
        for i in range(LOOKBACK_WINDOW, len(df)):
            # Use last LOOKBACK_WINDOW values as features
            feature_window = df['value'].iloc[i-LOOKBACK_WINDOW:i].values
            target = df['value'].iloc[i]
            
            # Add time-based features
            timestamp = df['timestamp'].iloc[i]
            hour = timestamp.hour
            day_of_week = timestamp.weekday()
            day_of_year = timestamp.timetuple().tm_yday
            
            # Combine all features
            feature_vector = np.concatenate([
                feature_window,
                [hour, day_of_week, day_of_year]
            ])
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)

    def train_model(self, X, y):
        """Train the time-series prediction model"""
        logger.info(f"Training model with {len(X)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model - using RandomForest for robustness
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        metrics = {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X.shape[1],
            'retrained_at': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Model trained - Test MAE: {test_mae:.3f}, Test RMSE: {test_rmse:.3f}")
        
        return model, scaler, metrics

    def save_model(self, model, scaler, metrics):
        """Save trained model and metadata"""
        # Ensure model directory exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Save model artifacts
        model_data = {
            'model': model,
            'scaler': scaler,
            'lookback_window': LOOKBACK_WINDOW,
            'metrics': metrics,
            'version': f"v{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        }
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Store metadata in Redis for monitoring
        if self.redis_client:
            try:
                metadata_key = "model:metadata"
                self.redis_client.set(metadata_key, json.dumps(metrics))
                
                # Store retraining history
                history_key = f"model:history:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                self.redis_client.setex(history_key, 86400 * 7, json.dumps(metrics))  # Keep for 7 days
                
                logger.info("Model metadata stored in Redis")
            except Exception as e:
                logger.warning(f"Failed to store metadata in Redis: {e}")

    def save_training_data(self, df):
        """Save training data to persistent storage for future reference"""
        data_file = os.path.join(DATA_PATH, f'training_data_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.csv')
        os.makedirs(DATA_PATH, exist_ok=True)
        
        df.to_csv(data_file, index=False)
        logger.info(f"Training data snapshot saved to {data_file}")

    def run_retraining(self):
        """Main retraining workflow"""
        logger.info("Starting model retraining workflow")
        
        try:
            # 1. Load historical data from persistent storage
            historical_df = self.load_historical_data()
            
            # 2. Load recent data from Redis cache
            recent_df = self.load_recent_data_from_redis()
            
            # 3. Combine datasets
            if not recent_df.empty:
                # Remove any overlap and combine
                combined_df = pd.concat([historical_df, recent_df]).drop_duplicates(
                    subset=['timestamp'], keep='last'
                ).sort_values('timestamp')
                logger.info(f"Combined {len(historical_df)} historical + {len(recent_df)} recent data points")
            else:
                combined_df = historical_df
                logger.info(f"Using {len(historical_df)} historical data points only")
            
            # 4. Prepare features
            X, y = self.prepare_features(combined_df)
            
            if len(X) < 100:
                logger.error("Insufficient data for training (need at least 100 samples)")
                return False
            
            # 5. Train model
            model, scaler, metrics = self.train_model(X, y)
            
            # 6. Save model and metadata
            self.save_model(model, scaler, metrics)
            
            # 7. Save training data snapshot
            self.save_training_data(combined_df)
            
            logger.info("Retraining completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return False

def main():
    """Main entry point for retraining script"""
    logger.info("Time-Series Model Retrainer Starting")
    
    retrainer = TimeSeriesRetrainer()
    success = retrainer.run_retraining()
    
    if success:
        logger.info("Retraining completed successfully")
        sys.exit(0)
    else:
        logger.error("Retraining failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
