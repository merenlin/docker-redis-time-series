#!/bin/bash
echo "🧪 Quick Workshop Test"
echo "====================="

echo "1. Starting services..."
cd .. && docker-compose up -d && cd workshop-files

echo "2. Waiting for services to be ready..."
sleep 10

echo "3. Testing health endpoint..."
curl -s http://localhost:5001/health | jq .

echo "4. Testing prediction endpoint..."
curl -s -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "series_id": "workshop-test",
    "historical_data": [
      {"timestamp": "2025-01-01T00:00:00", "value": 100.0},
      {"timestamp": "2025-01-01T01:00:00", "value": 101.0}
    ]
  }' | jq .

echo "5. Checking Redis data..."
cd .. && docker-compose exec redis redis-cli KEYS "*workshop-test*" && cd workshop-files

echo "✅ Quick test completed!"
echo "🌐 API available at: http://localhost:5001"
echo "📚 API docs at: http://localhost:5001/"
