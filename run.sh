#!/bin/bash

# Personal Finance Agent - Quick Start Script

echo "Personal Finance Agent"
echo "====================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "✅ Created .env file"
    echo "📝 Please edit .env and add your API keys before running the app"
    echo ""
    read -p "Press Enter after you've configured your API keys..."
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Starting Streamlit app..."
echo ""

# Run the app
streamlit run app.py
