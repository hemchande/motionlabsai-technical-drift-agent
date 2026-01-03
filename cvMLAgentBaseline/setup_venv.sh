#!/bin/bash
# Setup script for CV ML Agent virtual environment

set -e

echo "🔧 Setting up CV ML Agent virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "✅ Virtual environment created/verified"
echo "📥 Installing dependencies..."

# Upgrade pip
.venv/bin/pip install --upgrade pip setuptools wheel

# Install core dependencies
.venv/bin/pip install python-dotenv opencv-python numpy mediapipe ultralytics

# Optional: Install PyTorch (uncomment if needed)
# .venv/bin/pip install torch torchvision

# Optional: Install vision-agents (uncomment if needed)
# .venv/bin/pip install "vision-agents[getstream,openai,ultralytics,gemini]"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "Or use the Python directly:"
echo "  .venv/bin/python main.py"
echo ""






















