# Time-Series API with Redis State Management

A minimal Flask API for time-series predictions with Redis for state management, containerized with Docker.

## Quick Start

```bash
docker-compose up --build
```

## Test the API

```bash
# Health check
curl http://localhost:5000/health

# Make a prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1.0, 2.0, 3.0, 4.0, 5.0]}'
```

## Services

- **API**: Flask application on `http://localhost:5000`
- **Redis**: Cache and state storage on `localhost:6379`

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Time-series prediction
