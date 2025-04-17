# UFC Predictor Application Dockerfile
# A multi-stage build for both backend and frontend

# ---------------
# Backend Stage
# ---------------
FROM python:3.9-slim AS backend

WORKDIR /app/backend

# Install dependencies required for scraping and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Create necessary directories
RUN mkdir -p data/
RUN mkdir -p logs/
RUN mkdir -p models/

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# ---------------
# Frontend Stage
# ---------------
FROM node:16 AS frontend-build

WORKDIR /app/frontend

# Copy frontend dependencies
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci

# Copy frontend source
COPY frontend/ .

# Build frontend
RUN npm run build

# ---------------
# Final Stage
# ---------------
FROM python:3.9-slim

WORKDIR /app

# Install NGINX for serving frontend
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy backend from backend stage
COPY --from=backend /app/backend/ /app/backend/
COPY --from=backend /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=backend /usr/local/bin/ /usr/local/bin/
# Copy built frontend from frontend-build stage
COPY --from=frontend-build /app/frontend/build/ /app/frontend/build/

# Set up NGINX config for serving the frontend and proxying API requests
RUN echo 'server { \
    listen 80; \
    server_name _; \
    \
    # Serve frontend static files \
    location / { \
        root /app/frontend/build; \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Proxy API requests to the Flask backend \
    location /api/ { \
        proxy_pass http://127.0.0.1:5000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; \
    } \
}' > /etc/nginx/sites-available/default

# Create startup script
RUN echo '#!/bin/bash \n\
# Start Flask backend \n\
cd /app/backend \n\
gunicorn -w 4 -b 127.0.0.1:5000 main:app & \n\
\n\
# Start NGINX \n\
nginx -g "daemon off;"' > /app/start.sh

RUN chmod +x /app/start.sh

# Expose port 80
EXPOSE 80

# Set working directory to app root
WORKDIR /app

# Start services
CMD ["/app/start.sh"]