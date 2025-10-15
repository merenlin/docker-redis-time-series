"""
Test script for the Flask Time-Series API
This demonstrates how to use the various endpoints.
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import time

# API Configuration
API_BASE_URL = "http://localhost:5001"


def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def generate_test_data(n_points=100):
    """Generate test time-series data"""
    start_time = datetime.now() - timedelta(hours=n_points)

    data = []
    for i in range(n_points):
        timestamp = start_time + timedelta(hours=i)
        # Simple sine wave with trend and noise
        value = 100 + 0.1 * i + 10 * \
            np.sin(2 * np.pi * i / 24) + np.random.normal(0, 2)

        data.append({
            "timestamp": timestamp.isoformat(),
            "value": round(value, 2)
        })

    return data


def test_prediction_with_data():
    """Test prediction endpoint with historical data"""
    print("\n📈 Testing prediction with historical data...")

    try:
        import numpy as np
        historical_data = generate_test_data(80)

        payload = {
            "series_id": "test_series_1",
            "historical_data": historical_data
        }

        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False


def test_add_data():
    """Test adding new data points"""
    print("\n📊 Testing add data endpoint...")

    try:
        # Generate new data points
        new_data = []
        for i in range(5):
            timestamp = datetime.now() + timedelta(hours=i)
            value = 120 + i * 0.5 + np.random.normal(0, 1)

            new_data.append({
                "timestamp": timestamp.isoformat(),
                "value": round(value, 2)
            })

        payload = {
            "series_id": "test_series_1",
            "data_points": new_data
        }

        response = requests.post(
            f"{API_BASE_URL}/add_data",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Add data test failed: {e}")
        return False


def test_prediction_with_cache():
    """Test prediction using cached data"""
    print("\n🗄️ Testing prediction with cached data...")

    try:
        payload = {
            "series_id": "test_series_1"
        }

        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Cached prediction test failed: {e}")
        return False


def test_series_status():
    """Test getting series status"""
    print("\n📋 Testing series status endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/status/test_series_1")

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False


def test_model_reload():
    """Test model reload endpoint"""
    print("\n🔄 Testing model reload endpoint...")

    try:
        response = requests.post(f"{API_BASE_URL}/reload_model")

        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")

        return response.status_code == 200

    except Exception as e:
        print(f"❌ Model reload test failed: {e}")
        return False


def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting Flask API Tests")
    print("=" * 50)

    tests = [
        ("Health Check", test_health_check),
        ("Prediction with Data", test_prediction_with_data),
        ("Add Data", test_add_data),
        ("Prediction with Cache", test_prediction_with_cache),
        ("Series Status", test_series_status),
        ("Model Reload", test_model_reload)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))

        time.sleep(1)  # Small delay between tests

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    # Import numpy here since it's only needed for testing
    import numpy as np

    print("Flask Time-Series API Test Suite")
    print("Make sure the Flask API is running on http://localhost:5001")
    print("You can start it with: python app.py")

    input("\nPress Enter to start tests...")
    run_all_tests()
