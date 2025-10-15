"""
Data ingestion service for continuous data flow.
Simulates real-world data streams feeding into the External State Store.
"""

import time
import json
import redis
import logging
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
DATA_PATH = os.getenv('DATA_PATH', '/app/data')
INGESTION_INTERVAL = int(os.getenv('INGESTION_INTERVAL', 30))  # seconds
LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', 20))

class DataIngester:
    def __init__(self):
        self.redis_client = None
        self.connect_redis()
        self.data_sources = ['sensor_1', 'sensor_2', 'website_traffic', 'sales_data']
        
    def connect_redis(self):
        """Connect to Redis for storing streaming data"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Data ingester connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def generate_realistic_data_point(self, series_id: str) -> Dict:
        """Generate realistic time-series data point"""
        now = datetime.utcnow()
        
        # Base patterns for different series types
        patterns = {
            'sensor_1': {
                'base': 25.0,  # Temperature sensor
                'trend': 0.001,
                'seasonal_amplitude': 5.0,
                'noise_level': 0.5
            },
            'sensor_2': {
                'base': 60.0,  # Humidity sensor
                'trend': -0.0005,
                'seasonal_amplitude': 10.0,
                'noise_level': 1.0
            },
            'website_traffic': {
                'base': 1000.0,
                'trend': 0.01,
                'seasonal_amplitude': 200.0,
                'noise_level': 50.0
            },
            'sales_data': {
                'base': 5000.0,
                'trend': 0.005,
                'seasonal_amplitude': 1000.0,
                'noise_level': 100.0
            }
        }
        
        pattern = patterns.get(series_id, patterns['sensor_1'])
        
        # Time-based features
        hour = now.hour
        day_of_week = now.weekday()
        
        # Generate value with realistic patterns
        value = pattern['base']
        
        # Add trend
        days_since_epoch = (now - datetime(2024, 1, 1)).days
        value += pattern['trend'] * days_since_epoch
        
        # Add daily seasonality
        value += pattern['seasonal_amplitude'] * np.sin(2 * np.pi * hour / 24)
        
        # Add weekly seasonality
        value += pattern['seasonal_amplitude'] * 0.3 * np.sin(2 * np.pi * day_of_week / 7)
        
        # Add noise
        value += np.random.normal(0, pattern['noise_level'])
        
        return {
            'timestamp': now.isoformat(),
            'value': round(float(value), 2),
            'series_id': series_id,
            'source': 'data_ingester'
        }

    def store_data_point(self, data_point: Dict):
        """Store data point in Redis using time-based keys for scalability"""
        series_id = data_point['series_id']
        timestamp = data_point['timestamp']
        
        try:
            # Store individual data point with time-based key for easy retrieval
            point_key = f"timeseries:{series_id}:{timestamp}"
            redis_data = {
                'timestamp': timestamp,
                'value': data_point['value'],
                'series_id': series_id
            }
            
            # Store individual point (no expiry for persistent storage)
            self.redis_client.set(point_key, json.dumps(redis_data))
            
            # Maintain recent data cache for fast predictions (last LOOKBACK_WINDOW points)
            cache_key = f"recent:{series_id}"
            existing_recent = []
            
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                existing_recent = json.loads(cached_data)
            
            # Add new point to recent cache
            existing_recent.append({
                'timestamp': timestamp,
                'value': data_point['value']
            })
            
            # Keep only recent points for fast access
            recent_data = existing_recent[-LOOKBACK_WINDOW:]
            self.redis_client.set(cache_key, json.dumps(recent_data))
            
            # Add to series index for easy discovery
            self.redis_client.sadd("series:active", series_id)
            
            # Add timestamp to series timeline for efficient range queries
            timestamp_score = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
            self.redis_client.zadd(f"timeline:{series_id}", {timestamp: timestamp_score})
            
            logger.info(f"Stored data point for {series_id}: {data_point['value']} at {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to store data point: {e}")

    def update_ingestion_metrics(self):
        """Update ingestion metrics in Redis"""
        try:
            metrics_key = "ingestion:metrics"
            current_time = datetime.utcnow().isoformat()
            
            # Get current metrics
            current_metrics = {}
            existing = self.redis_client.get(metrics_key)
            if existing:
                current_metrics = json.loads(existing)
            
            # Update metrics
            current_metrics.update({
                'last_ingestion': current_time,
                'total_points_today': current_metrics.get('total_points_today', 0) + len(self.data_sources),
                'status': 'active',
                'sources': self.data_sources,
                'storage_type': 'redis_only'
            })
            
            self.redis_client.setex(metrics_key, 86400, json.dumps(current_metrics))
            
        except Exception as e:
            logger.error(f"Failed to update ingestion metrics: {e}")

    def run_continuous_ingestion(self):
        """Main ingestion loop"""
        logger.info(f"Starting continuous data ingestion every {INGESTION_INTERVAL} seconds")
        logger.info(f"Data sources: {self.data_sources}")
        
        try:
            while True:
                # Generate and store data for each source
                for series_id in self.data_sources:
                    data_point = self.generate_realistic_data_point(series_id)
                    self.store_data_point(data_point)
                
                # Update metrics
                self.update_ingestion_metrics()
                
                # Wait for next ingestion cycle
                time.sleep(INGESTION_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("Data ingestion stopped by user")
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise

    def run_batch_ingestion(self, num_points: int = 100):
        """Run batch ingestion for testing purposes"""
        logger.info(f"Starting batch ingestion of {num_points} points per series")
        
        for series_id in self.data_sources:
            logger.info(f"Generating data for {series_id}")
            
            for i in range(num_points):
                # Generate historical data points
                base_time = datetime.utcnow() - timedelta(hours=num_points-i)
                data_point = self.generate_realistic_data_point(series_id)
                data_point['timestamp'] = base_time.isoformat()
                
                self.store_data_point(data_point)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"Generated {i+1}/{num_points} points for {series_id}")
        
        logger.info("Batch ingestion completed")

def main():
    """Main entry point"""
    import sys
    
    logger.info("Time-Series Data Ingester Starting")
    
    ingester = DataIngester()
    
    # Check if running in batch mode
    if len(sys.argv) > 1 and sys.argv[1] == '--batch':
        num_points = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        ingester.run_batch_ingestion(num_points)
    else:
        ingester.run_continuous_ingestion()

if __name__ == "__main__":
    main()
