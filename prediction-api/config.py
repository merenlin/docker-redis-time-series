"""
Configuration settings for the Flask Time-Series API
"""

import os


class Config:
    # Redis Settings
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_TTL_TIMESERIES = int(
        os.getenv('REDIS_TTL_TIMESERIES', 3600))  # 1 hour
    REDIS_TTL_PREDICTIONS = int(
        os.getenv('REDIS_TTL_PREDICTIONS', 86400))  # 24 hours

    # Model Settings
    MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/timeseries_model.pkl')
    LOOKBACK_WINDOW = int(os.getenv('LOOKBACK_WINDOW', 20))
    DATA_PATH = os.getenv('DATA_PATH', '/app/data')

    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 'yes']
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5001))

    # API Settings
    MAX_DATA_POINTS_PER_REQUEST = int(
        os.getenv('MAX_DATA_POINTS_PER_REQUEST', 1000))
    MAX_SERIES_PER_INSTANCE = int(os.getenv('MAX_SERIES_PER_INSTANCE', 100))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Kubernetes/Production Settings
    WORKERS = int(os.getenv('GUNICORN_WORKERS', 2))
    TIMEOUT = int(os.getenv('GUNICORN_TIMEOUT', 60))
    MAX_REQUESTS = int(os.getenv('GUNICORN_MAX_REQUESTS', 1000))
    MAX_REQUESTS_JITTER = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 100))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


class TestingConfig(Config):
    TESTING = True
    REDIS_DB = 1  # Use different DB for testing


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
