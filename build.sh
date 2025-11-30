#!/usr/bin/env bash
# Build script for Render.com

set -o errexit  # Exit on error

echo "🔧 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build completed successfully!"
