#!/bin/bash
echo "Restarting UFC server..."

# Find and kill any running gunicorn processes
for pid in $(ps -ef | grep "gunicorn\|main.py" | grep -v grep | awk '{print $2}'); do
  echo "Killing process $pid"
  kill -9 $pid
done

cd /home/ubuntu/ufc-app/ufc-predictor
echo "Starting server in background..."
nohup python3 backend/main.py server > server.log 2>&1 &

echo "Server restarted, waiting for startup..."
sleep 3
echo "Testing API health endpoint..."
curl -X GET http://localhost:5000/api/health
echo ""
echo "Restart completed"