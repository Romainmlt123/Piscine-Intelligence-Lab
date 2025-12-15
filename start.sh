#!/bin/bash

# Voice Agent Startup Script

# Activate virtual environment
source venv/bin/activate

# Set Python path to include src
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Start the server
echo "Starting Voice Agent Server..."
python src/main.py
