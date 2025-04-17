#!/bin/bash
echo "Restarting UFC server..."

# Find and kill any running gunicorn processes
for pid in $(ps -ef | grep "gunicorn\|main.py" | grep -v grep | awk '{print $2}'); do
  echo "Killing process $pid"
  kill -9 $pid
done

cd /home/ubuntu/ufc-app/ufc-predictor

# Get the port from .env file with a default of 5001
PORT=5001
if [[ -f "backend/.env" ]]; then
  ENV_PORT=$(grep "PORT=" backend/.env | cut -d= -f2)
  if [[ ! -z "$ENV_PORT" ]]; then
    PORT=$ENV_PORT
  fi
fi

echo "Starting server in background on port $PORT..."
nohup python3 backend/main.py server > server.log 2>&1 &

echo "Server restarted, waiting for startup..."
sleep 3
echo "Testing API health endpoint..."
curl -X GET http://localhost:$PORT/api/health
echo ""
echo "Restart completed"