#!/bin/bash

# Workshop Setup Script
# This script prepares the environment for the Redis workshop

set -e

echo "🚀 Setting up Redis Workshop Environment..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker Desktop."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose."
    exit 1
fi

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if Kubernetes is enabled
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Kubernetes cluster is not accessible. Please enable Kubernetes in Docker Desktop."
    exit 1
fi

echo "✅ All prerequisites are met!"

# Build the Docker image
echo "🐳 Building Docker image..."
cd ../prediction-api && docker build -t prediction-api:latest . && cd ../workshop-files

# Copy important files to current directory (we're already in workshop-files)
echo "📋 Preparing workshop files..."
# Files are already in the right place since we're running from workshop-files
# Note: docker-compose.yml stays in prediction-api directory where it belongs

# Copy instructor notes if they exist
if [ -f "../prediction-api/workshop-instructor-notes.md" ]; then
    cp ../prediction-api/workshop-instructor-notes.md .
fi

# Create a quick test script
cat > quick-test.sh << 'EOF'
#!/bin/bash
echo "🧪 Quick Workshop Test"
echo "====================="

echo "1. Starting services..."
docker-compose up -d

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
docker-compose exec redis redis-cli KEYS "*workshop-test*"

echo "✅ Quick test completed!"
echo "🌐 API available at: http://localhost:5001"
echo "📚 API docs at: http://localhost:5001/"
EOF

chmod +x quick-test.sh

# Create cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash
echo "🧹 Cleaning up workshop environment..."

echo "Stopping Docker Compose services..."
cd ../prediction-api && docker-compose down -v && cd ../workshop-files

echo "Cleaning up Kubernetes resources..."
kubectl delete namespace prediction-api 2>/dev/null || true

echo "✅ Cleanup completed!"
EOF

chmod +x cleanup.sh

echo ""
echo "🎉 Workshop setup completed!"
echo ""
echo "📁 Workshop files are ready in current directory"
echo "🧪 Quick test: ./quick-test.sh"
echo "🧹 Cleanup: ./cleanup.sh"
echo ""
echo "📚 Workshop guide: ./workshop.md"
echo ""
echo "🚀 Ready for the workshop!"
