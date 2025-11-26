#!/bin/bash
# Start Ollama server in background if not running
if ! pgrep -x "ollama" > /dev/null
then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Activate venv
source venv/bin/activate

# Run server
echo "Starting Voice Agent Server..."
python server.py
