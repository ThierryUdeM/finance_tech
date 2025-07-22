#!/usr/bin/env python3
"""
Initialize technical signal files in Azure Storage
This script creates initial empty signal files for the Shiny app to load
"""

import os
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# Load Azure credentials
load_dotenv('config/.env')

def initialize_azure_files():
    """Create initial technical signal files in Azure"""
    
    # Azure configuration
    storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
    storage_key = os.getenv('AZURE_STORAGE_KEY')
    container_name = os.getenv('AZURE_CONTAINER_NAME')
    
    if not all([storage_account, storage_key, container_name]):
        print("Azure credentials not found in config/.env")
        return False
    
    # Initialize Azure client
    blob_service_client = BlobServiceClient(
        account_url=f"https://{storage_account}.blob.core.windows.net",
        credential=storage_key
    )
    
    try:
        # Create initial same-day technical signal
        current_signal = {
            "ticker": "NVDA",
            "signal": "HOLD",
            "confidence": 0.5,
            "price": 0.0,
            "components": {
                "ma_cross": 0,
                "rsi": 0,
                "stoch": 0,
                "bb": 0,
                "macd": 0,
                "volume": 0,
                "momentum": 0
            },
            "weighted_score": 0.0,
            "indicators": {
                "rsi": None,
                "stoch_k": None,
                "macd_hist": None,
                "volume_ratio": None,
                "atr": None,
                "volatility_ratio": 1.0
            },
            "stop_loss": None,
            "take_profit": None,
            "interval": "15m",
            "scan_time": datetime.now().isoformat(),
            "expiry_time": datetime.now().isoformat(),
            "strength": "Neutral",
            "message": "No signals generated yet. Run the workflow to generate signals."
        }
        
        # Upload current signal
        blob_name = "same_day_technical/current_signal.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        json_data = json.dumps(current_signal, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"Created {blob_name}")
        
        # Create initial performance summary
        performance_summary = {
            "total_signals": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_confidence": 0.5,
            "hour_distribution": {},
            "last_updated": datetime.now().isoformat(),
            "message": "No performance data yet. Signals will be tracked once workflow runs."
        }
        
        blob_name = "same_day_technical/performance_summary.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        json_data = json.dumps(performance_summary, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"Created {blob_name}")
        
        # Create empty evaluations file
        evaluations = []
        blob_name = "same_day_technical/technical_evaluations.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        json_data = json.dumps(evaluations, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"Created {blob_name}")
        
        # Create initial next-day predictions
        next_day_predictions = {
            "ticker": "NVDA",
            "scan_date": datetime.now().isoformat(),
            "market_close_date": datetime.now().date().isoformat(),
            "predictions": [],
            "summary": {
                "total_patterns": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "recommendation": "HOLD",
                "confidence": 0.0,
                "message": "No patterns detected yet. Daily scanner runs at 3:55 PM ET."
            }
        }
        
        blob_name = "next_day_technical/next_day_predictions.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        json_data = json.dumps(next_day_predictions, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"Created {blob_name}")
        
        # Create empty pattern evaluations
        pattern_evaluations = []
        blob_name = "next_day_technical/pattern_evaluations.json"
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        json_data = json.dumps(pattern_evaluations, indent=2)
        blob_client.upload_blob(json_data, overwrite=True)
        print(f"Created {blob_name}")
        
        print("\nAll initial files created successfully!")
        print("\nNext steps:")
        print("1. Deploy the workflows to GitHub")
        print("2. The simple technical scanner will run every 15 minutes during market hours")
        print("3. The daily pattern scanner will run at 3:55 PM ET")
        print("4. The Shiny app will now display 'No Signal Yet' instead of 'Loading...'")
        
        return True
        
    except Exception as e:
        print(f"Error creating files: {e}")
        return False


if __name__ == "__main__":
    initialize_azure_files()