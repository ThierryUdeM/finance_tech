#!/usr/bin/env python3
"""
Run all 4 momentum+shape models for live predictions
Saves combined results to Azure
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import yfinance as yf
from azure.storage.blob import BlobServiceClient
import warnings
warnings.filterwarnings('ignore')

# Add paths - handle both local and GitHub Actions environments
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

# Try to find the models in different locations
model_paths = [
    os.path.join(repo_root, 'walk_forward_tests', 'model', 'momentum_shapematching'),
    os.path.join(repo_root, 'model', 'momentum_shapematching'),
    os.path.join(script_dir, '..', 'walk_forward_tests', 'model', 'momentum_shapematching')
]

for path in model_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        print(f"Found models at: {path}")
        break
else:
    print("Warning: Could not find model directory, trying direct import...")

# Import models
try:
    from nvda_v1 import nvda_v1_model
    from aapl_improved import aapl_improved_model
    from msft_improved import msft_improved_model
    from v1_TSLA import v1_TSLA_model
except ImportError as e:
    print(f"Error importing models: {e}")
    print("Current sys.path:")
    for p in sys.path[:5]:
        print(f"  {p}")
    raise

# Model mapping
MODELS = {
    'NVDA': ('nvda_v1', nvda_v1_model),
    'AAPL': ('aapl_improved', aapl_improved_model),
    'MSFT': ('msft_improved', msft_improved_model),
    'TSLA': ('v1_TSLA', v1_TSLA_model)
}

def initialize_azure():
    """Initialize Azure blob storage client"""
    account_name = os.environ.get('STORAGE_ACCOUNT_NAME')
    access_key = os.environ.get('ACCESS_KEY')
    container_name = os.environ.get('CONTAINER_NAME')
    
    if not all([account_name, access_key, container_name]):
        raise ValueError("Azure credentials not found in environment variables")
    
    blob_service_client = BlobServiceClient(
        account_url=f"https://{account_name}.blob.core.windows.net",
        credential=access_key
    )
    
    return blob_service_client.get_container_client(container_name)

def fetch_recent_data(ticker, lookback_days=90):
    """Fetch recent data from yfinance"""
    print(f"\nFetching {lookback_days} days of data for {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    try:
        # Download 15-minute data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='15m',
            progress=False
        )
        
        if data.empty:
            print(f"No data received for {ticker}")
            return None
            
        # Prepare data format expected by models
        data.reset_index(inplace=True)
        data.columns = ['ts_event', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        data = data[['ts_event', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"  Fetched {len(data)} bars from {data['ts_event'].min()} to {data['ts_event'].max()}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def generate_predictions(ticker, model_name, model_func, data):
    """Generate predictions using the specified model"""
    print(f"\nGenerating predictions for {ticker} using {model_name}...")
    
    # Split data for train/test (use last 60 days for training, predict on latest data)
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    # Ensure index is datetime for the models
    train_data.set_index('ts_event', inplace=True)
    test_data.set_index('ts_event', inplace=True)
    
    try:
        # Run the model
        signals = model_func(train_data, test_data)
        
        # Get the latest signal
        latest_signals = signals[signals['signal'] != 0].tail(5)
        
        if len(latest_signals) > 0:
            print(f"  Found {len(latest_signals)} recent signals")
            
            predictions = []
            for idx, row in latest_signals.iterrows():
                pred = {
                    'timestamp': idx.isoformat(),
                    'signal': int(row['signal']),
                    'price': float(test_data.loc[idx, 'close']),
                    'model': model_name
                }
                predictions.append(pred)
                print(f"    {idx}: {'BUY' if row['signal'] > 0 else 'SELL'} @ ${pred['price']:.2f}")
            
            return predictions
        else:
            print("  No signals generated")
            return []
            
    except Exception as e:
        print(f"  Error running model: {str(e)}")
        return []

def main():
    """Main execution function"""
    print("="*80)
    print("MOMENTUM+SHAPE MODELS - LIVE PREDICTIONS")
    print("="*80)
    
    # Initialize Azure
    try:
        container_client = initialize_azure()
        print("✓ Azure connection established")
    except Exception as e:
        print(f"✗ Azure connection failed: {str(e)}")
        return
    
    # Results storage
    all_results = {
        'run_timestamp': datetime.utcnow().isoformat() + 'Z',
        'models_used': list(MODELS.keys()),
        'predictions': {}
    }
    
    # Run each model
    for ticker, (model_name, model_func) in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {ticker}")
        print(f"{'='*60}")
        
        # Fetch data
        data = fetch_recent_data(ticker)
        if data is None:
            print(f"Skipping {ticker} - no data available")
            all_results['predictions'][ticker] = {
                'status': 'error',
                'error': 'No data available',
                'model': model_name
            }
            continue
        
        # Generate predictions
        predictions = generate_predictions(ticker, model_name, model_func, data)
        
        # Store results
        all_results['predictions'][ticker] = {
            'status': 'success',
            'model': model_name,
            'data_points': len(data),
            'latest_price': float(data['close'].iloc[-1]),
            'latest_timestamp': data['ts_event'].iloc[-1].isoformat(),
            'signals': predictions,
            'signal_count': len(predictions)
        }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_signals = sum(
        result.get('signal_count', 0) 
        for result in all_results['predictions'].values() 
        if result.get('status') == 'success'
    )
    
    print(f"Total signals generated: {total_signals}")
    for ticker, result in all_results['predictions'].items():
        if result['status'] == 'success':
            print(f"  {ticker}: {result['signal_count']} signals using {result['model']}")
        else:
            print(f"  {ticker}: Error - {result.get('error', 'Unknown')}")
    
    # Save to Azure
    try:
        # Create filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        blob_name = f"predictions/momentum_models_{timestamp}.json"
        
        # Upload
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            json.dumps(all_results, indent=2),
            overwrite=True
        )
        
        print(f"\n✓ Results saved to Azure: {blob_name}")
        
        # Also save a "latest" version for easy access
        latest_blob = container_client.get_blob_client("predictions/momentum_models_latest.json")
        latest_blob.upload_blob(
            json.dumps(all_results, indent=2),
            overwrite=True
        )
        print("✓ Latest results updated")
        
    except Exception as e:
        print(f"\n✗ Failed to save to Azure: {str(e)}")
        # Save locally as backup
        local_file = f"momentum_predictions_{timestamp}.json"
        with open(local_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Results saved locally: {local_file}")
    
    print("\nComplete!")

if __name__ == "__main__":
    main()