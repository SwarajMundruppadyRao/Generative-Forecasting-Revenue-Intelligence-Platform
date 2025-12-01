#!/bin/bash
# Quick start script for the Revenue Intelligence Platform

echo "🚀 Revenue Intelligence Platform - Quick Start"
echo "=============================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
fi

# Start Neo4j with Docker
echo "🗄️  Starting Neo4j..."
cd docker
docker-compose up -d neo4j
cd ..

# Wait for Neo4j to be ready
echo "⏳ Waiting for Neo4j to start..."
sleep 10

# Run setup (skip training for quick start)
echo "🔨 Running setup (without training)..."
python main.py --skip-training

echo ""
echo "✅ Quick start complete!"
echo ""
echo "To start the API server:"
echo "  python -m api.server"
echo ""
echo "Or with Docker:"
echo "  cd docker && docker-compose up -d"
echo ""
echo "API Documentation: http://localhost:8000/docs"
