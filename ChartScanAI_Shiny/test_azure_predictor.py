#!/usr/bin/env python3
"""
Test script to check Azure connection and run a single prediction
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../config/.env')

# Check if environment variables are loaded
print("Environment Variables Check:")
print(f"AZURE_STORAGE_ACCOUNT: {os.getenv('AZURE_STORAGE_ACCOUNT')}")
print(f"AZURE_CONTAINER_NAME: {os.getenv('AZURE_CONTAINER_NAME')}")
print(f"AZURE_STORAGE_KEY: {'***' if os.getenv('AZURE_STORAGE_KEY') else 'NOT FOUND'}")

# Try to run the predictor
try:
    from btc_predictor_azure import BTCPredictor
    
    print("\nInitializing predictor...")
    predictor = BTCPredictor()
    
    print("Running prediction...")
    result = predictor.run_prediction()
    
    if result:
        print(f"Prediction successful! Uploaded to: {result}")
    else:
        print("Prediction failed")
        
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()