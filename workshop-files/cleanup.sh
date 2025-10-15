#!/bin/bash
echo "🧹 Cleaning up workshop environment..."

echo "Stopping Docker Compose services..."
cd ../prediction-api && docker-compose down -v && cd ../workshop-files

echo "Cleaning up Kubernetes resources..."
kubectl delete namespace prediction-api 2>/dev/null || true

echo "✅ Cleanup completed!"
