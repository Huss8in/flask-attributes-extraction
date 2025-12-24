#!/bin/bash
# Quick start script for Docker services

set -e

echo "======================================================================"
echo "Flask Attribute Extractor - Docker Quick Start"
echo "======================================================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "[ERROR] .env file not found!"
    echo "[INFO] Creating .env from .env.example..."
    cp .env.example .env
    echo "[WARNING] Please edit .env and add your API keys:"
    echo "  - OPENAI_API_KEY"
    echo "  - API_URL (for Ollama/Aya translation)"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if GradProject/.env exists
if [ ! -f GradProject/.env ]; then
    echo "[ERROR] GradProject/.env file not found!"
    echo "[INFO] Creating GradProject/.env from .env.example..."
    cp GradProject/.env.example GradProject/.env
    echo "[WARNING] Please edit GradProject/.env and add your HF_TOKEN"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] Docker Compose is not installed or not in PATH"
    exit 1
fi

# Check for NVIDIA Docker (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "[WARNING] NVIDIA Docker runtime not available or GPU not detected"
    echo "[WARNING] GradProject service requires GPU and will fail to start"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "======================================================================"
echo "Starting services..."
echo "======================================================================"
echo ""

# Build and start services
docker-compose up --build -d

echo ""
echo "======================================================================"
echo "Services started successfully!"
echo "======================================================================"
echo ""
echo "Access points:"
echo "  - Flask API:     http://localhost:6002"
echo "  - GradProject:   http://localhost:5000"
echo "  - Health Check:  http://localhost:6002/health"
echo ""
echo "Useful commands:"
echo "  - View logs:     docker-compose logs -f"
echo "  - Stop services: docker-compose down"
echo "  - Restart:       docker-compose restart"
echo ""
echo "======================================================================"
