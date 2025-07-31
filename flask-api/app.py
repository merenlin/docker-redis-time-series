from flask import Flask, request, jsonify
import redis
import numpy as np
import pandas as pd
import pickle
import json
import logging
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Redis configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Model configuration
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/timeseries_model.pkl')
# Number of previous points needed for scikit-learn model
LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', 5))

# Initialize Redis connection
try:
    redis_client = redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None


class TimeSeriesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()

    def load_model(self):
        """Load the trained model from disk"""
        try:
            if os.path.exists(MODEL_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.lookback_window = model_data.get('lookback_window', 5)
                logger.info(f"Model loaded successfully from {MODEL_PATH}")
            else:
                logger.warning(
                    f"Model file not found at {MODEL_PATH}. Using dummy model.")
                self._create_dummy_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self._create_dummy_model()

    def _create_dummy_model(self):
        """Create a simple dummy model for demonstration purposes"""
        class DummyModel:
            def predict(self, X):
                # Simple trend + noise prediction for 2D scikit-learn format
                if len(X.shape) == 2:
                    # Get last value from each sequence (last feature)
                    last_values = X[:, -1]
                    # Small upward trend with noise
                    trend = np.random.normal(0.02, 0.1, len(last_values))
                    return last_values * (1 + trend)
                else:
                    return np.array([100.0])  # Fallback

        self.model = DummyModel()
        self.scaler = None
        self.lookback_window = 5
        logger.info("Using dummy model for demonstration")

    def predict(self, historical_data: List[Dict]) -> float:
        """Make a prediction based on historical data"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            # Extract values
            values = df['value'].values.astype(float)

            # Get lookback window (use instance variable or fallback)
            window = getattr(self, 'lookback_window', LOOKBACK_WINDOW)

            # Prepare features (last 'window' values)
            if len(values) >= window:
                features = values[-window:]
            else:
                # Pad with the mean if not enough data
                mean_val = np.mean(values) if len(values) > 0 else 100.0
                features = np.full(window, mean_val)
                features[-len(values):] = values

            # Reshape for scikit-learn (1 sample, window features)
            X = features.reshape(1, -1)

            # Apply scaling if available
            if self.scaler:
                X = self.scaler.transform(X)

            # Make prediction
            prediction = self.model.predict(X)

            return float(prediction[0])

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(traceback.format_exc())
            raise


# Initialize predictor
predictor = TimeSeriesPredictor()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'redis_connected': redis_client is not None,
        'model_loaded': predictor.model is not None
    }

    if redis_client:
        try:
            redis_client.ping()
            status['redis_status'] = 'connected'
        except:
            status['redis_status'] = 'disconnected'
            status['redis_connected'] = False

    return jsonify(status)


@app.route('/predict', methods=['POST'])
def predict():
    """Make a time-series prediction"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Get series_id for this time series (default to 'default')
        series_id = data.get('series_id', 'default')

        # Handle two modes: with historical data or using cached data
        if 'historical_data' in data:
            # Mode 1: Historical data provided in request
            historical_data = data['historical_data']

            # Validate historical data format
            if not isinstance(historical_data, list) or len(historical_data) == 0:
                return jsonify({'error': 'historical_data must be a non-empty list'}), 400

            # Cache this data in Redis for future use
            if redis_client:
                try:
                    redis_key = f"timeseries:{series_id}"
                    # Store last N points for future predictions
                    cache_data = historical_data[-LOOKBACK_WINDOW:]
                    redis_client.setex(redis_key, 3600, json.dumps(
                        cache_data))  # Cache for 1 hour
                    logger.info(
                        f"Cached {len(cache_data)} data points for series {series_id}")
                except Exception as e:
                    logger.warning(f"Failed to cache data in Redis: {e}")

        elif redis_client:
            # Mode 2: Use cached historical data from Redis
            try:
                redis_key = f"timeseries:{series_id}"
                cached_data = redis_client.get(redis_key)

                if not cached_data:
                    return jsonify({'error': f'No historical data found for series_id: {series_id}. Please provide historical_data in the request.'}), 400

                historical_data = json.loads(cached_data)
                logger.info(
                    f"Retrieved {len(historical_data)} cached data points for series {series_id}")

            except Exception as e:
                logger.error(f"Error retrieving data from Redis: {e}")
                return jsonify({'error': 'Failed to retrieve historical data from cache'}), 500

        else:
            return jsonify({'error': 'No Redis connection and no historical_data provided'}), 500

        # Make prediction
        try:
            prediction_value = predictor.predict(historical_data)

            # Prepare response
            response = {
                'prediction': prediction_value,
                'series_id': series_id,
                'timestamp': datetime.utcnow().isoformat(),
                'data_points_used': len(historical_data),
                'model_type': 'LSTM' if hasattr(predictor.model, 'layers') else 'dummy'
            }

            # Store prediction in Redis for monitoring
            if redis_client:
                try:
                    prediction_key = f"predictions:{series_id}:{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                    redis_client.setex(prediction_key, 86400, json.dumps(
                        response))  # Keep for 24 hours
                except Exception as e:
                    logger.warning(f"Failed to store prediction in Redis: {e}")

            logger.info(
                f"Prediction made for series {series_id}: {prediction_value}")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Request processing error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Request processing failed: {str(e)}'}), 500


@app.route('/add_data', methods=['POST'])
def add_data():
    """Add new data points to the time series"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        series_id = data.get('series_id', 'default')
        new_data_points = data.get('data_points', [])

        if not isinstance(new_data_points, list) or len(new_data_points) == 0:
            return jsonify({'error': 'data_points must be a non-empty list'}), 400

        if not redis_client:
            return jsonify({'error': 'Redis not available for data storage'}), 500

        # Get existing data
        redis_key = f"timeseries:{series_id}"
        existing_data = []

        try:
            cached_data = redis_client.get(redis_key)
            if cached_data:
                existing_data = json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Could not retrieve existing data: {e}")

        # Combine and keep only the last LOOKBACK_WINDOW points
        all_data = existing_data + new_data_points
        recent_data = all_data[-LOOKBACK_WINDOW:]

        # Store back to Redis
        try:
            redis_client.setex(redis_key, 3600, json.dumps(recent_data))

            response = {
                'status': 'success',
                'series_id': series_id,
                'points_added': len(new_data_points),
                'total_points_stored': len(recent_data),
                'timestamp': datetime.utcnow().isoformat()
            }

            logger.info(
                f"Added {len(new_data_points)} data points to series {series_id}")
            return jsonify(response)

        except Exception as e:
            logger.error(f"Failed to store data in Redis: {e}")
            return jsonify({'error': f'Failed to store data: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"Add data error: {e}")
        return jsonify({'error': f'Failed to add data: {str(e)}'}), 500


@app.route('/status/<series_id>', methods=['GET'])
def get_series_status(series_id):
    """Get status information for a specific time series"""
    try:
        if not redis_client:
            return jsonify({'error': 'Redis not available'}), 500

        # Get cached data info
        redis_key = f"timeseries:{series_id}"
        cached_data = redis_client.get(redis_key)

        if not cached_data:
            return jsonify({
                'series_id': series_id,
                'status': 'no_data',
                'data_points': 0,
                'message': 'No data found for this series'
            })

        historical_data = json.loads(cached_data)

        # Get recent predictions
        prediction_pattern = f"predictions:{series_id}:*"
        prediction_keys = redis_client.keys(prediction_pattern)
        recent_predictions = len(prediction_keys)

        # Get data range
        if historical_data:
            timestamps = [point['timestamp']
                          for point in historical_data if 'timestamp' in point]
            if timestamps:
                earliest = min(timestamps)
                latest = max(timestamps)
            else:
                earliest = latest = None
        else:
            earliest = latest = None

        response = {
            'series_id': series_id,
            'status': 'active',
            'data_points': len(historical_data),
            'recent_predictions': recent_predictions,
            'data_range': {
                'earliest': earliest,
                'latest': latest
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Status check error: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/reload_model', methods=['POST'])
def reload_model():
    """Reload the model from disk (useful after retraining)"""
    try:
        global predictor
        predictor = TimeSeriesPredictor()

        return jsonify({
            'status': 'success',
            'message': 'Model reloaded successfully',
            'timestamp': datetime.utcnow().isoformat(),
            'model_loaded': predictor.model is not None
        })

    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return jsonify({'error': f'Model reload failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
