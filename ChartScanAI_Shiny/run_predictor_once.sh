#!/bin/bash
# Script to run the BTC predictor once to generate data for the Shiny app

echo "Running BTC Predictor to generate data for Azure..."

# Change to the script directory
cd "$(dirname "$0")"

# Check if virtual environment exists, if not create it
if [ ! -d "btc_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv btc_env
    source btc_env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source btc_env/bin/activate
fi

# Run the predictor
echo "Running predictor..."
python btc_predictor_azure.py

echo "Done! Check the Shiny app for updated data."