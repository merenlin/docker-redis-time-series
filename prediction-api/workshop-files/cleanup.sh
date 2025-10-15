#!/bin/bash
echo "🧹 Cleaning up workshop environment..."

echo "Stopping Docker Compose services..."
docker-compose down -v

echo "Cleaning up Kubernetes resources..."
kubectl delete namespace prediction-api 2>/dev/null || true

echo "Removing workshop files..."
cd .. && rm -rf workshop-files

echo "✅ Cleanup completed!"
