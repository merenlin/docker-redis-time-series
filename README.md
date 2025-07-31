# Simple Time-Series Prediction API 🚀

## What is this?

This is a simple web app that can predict future values based on your data. Think of it like asking "What's the next number in this sequence?"

For example, if you give it: `100, 101, 102, 103, 104`
It might predict the next number could be around: `105`

## What's inside?

- **Web API**: A simple website that accepts your data and gives predictions
- **Memory Storage**: Keeps track of previous predictions so it gets smarter over time
- **Everything runs in containers**: No messy setup - just one command and it works!

## How to run it?

### Step 1: Start everything

```bash
docker-compose up --build
```

Wait a few minutes for everything to start up. You'll see lots of text, but don't worry!

### Step 2: Check if it's working

Open your web browser or use this command:

```bash
curl http://localhost:5000/health
```

You should see something like: `"status": "healthy"`

### Step 3: Make a prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"historical_data": [
    {"timestamp": "2024-01-01", "value": 100.0},
    {"timestamp": "2024-01-02", "value": 101.0},
    {"timestamp": "2024-01-03", "value": 102.0},
    {"timestamp": "2024-01-04", "value": 103.0},
    {"timestamp": "2024-01-05", "value": 104.0}
  ]}'
```

## What you'll get back

The API will return something like:

```json
{
  "prediction": 105.2,
  "model_type": "dummy",
  "data_points_used": 5,
  "timestamp": "2025-07-31T13:05:09.122812"
}
```

This means: "Based on your 5 data points, I predict the next value will be around 105.2"

## What's running?

When you start the app, two things happen:

1. **Web API** starts on `http://localhost:5000` - this is where you send your data
2. **Redis Database** starts on port `6379` - this remembers things between requests

## How to stop it?

```bash
docker-compose down
```

## What can you do with this?

- Predict stock prices (though don't bet money on it!)
- Forecast website traffic
- Predict sales numbers
- Any data that changes over time!

## Need help?

- Make sure Docker is installed and running
- The first time might take a few minutes to download everything
- If something breaks, try `docker-compose down` then `docker-compose up --build` again

## Why is this useful?

This shows how to build a "smart" web service that:

- ✅ Remembers things (has memory)
- ✅ Can be easily moved between computers (containerized)
- ✅ Handles multiple requests at once
- ✅ Gives consistent results

Perfect for learning how modern web apps work!
