#!/usr/bin/env bash
set -e

echo "Installing Raspberry Pi dependencies..."

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip libportaudio2 portaudio19-dev

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p recordings

echo
echo "Installation complete."
echo "Run with:"
echo "  ./run.sh"
