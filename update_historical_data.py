#!/usr/bin/env python3
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import io

# Azure connection
account_name = os.environ.get('AZURE_STORAGE_ACCOUNT')
account_key = os.environ.get('AZURE_STORAGE_KEY')
container_name = os.environ.get('AZURE_CONTAINER_NAME')

if not (account_name and account_key and container_name):
    print('Azure credentials not found. Please set:')
    print('- STORAGE_ACCOUNT_NAME')
    print('- ACCESS_KEY')
    print('- CONTAINER_NAME')
    exit(1)

# Create connection
connection_string = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'
blob_service = BlobServiceClient.from_connection_string(connection_string)

# File names
historical_blob_name = 'NVDA/NVDA_15min_pattern_ready.csv'
backup_blob_name = f'NVDA/backups/NVDA_15min_pattern_ready_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

try:
    # Step 1: Download existing historical data from Azure
    print("Downloading existing historical data from Azure...")
    blob_client = blob_service.get_blob_client(container=container_name, blob=historical_blob_name)
    
    try:
        blob_data = blob_client.download_blob()
        historical_df = pd.read_csv(io.StringIO(blob_data.readall().decode('utf-8')), 
                                   index_col='timestamp', parse_dates=True)
        print(f"Loaded {len(historical_df)} existing records")
        print(f"Date range: {historical_df.index[0]} to {historical_df.index[-1]}")
        
        # Create backup
        backup_client = blob_service.get_blob_client(container=container_name, blob=backup_blob_name)
        backup_client.upload_blob(blob_data.readall(), overwrite=True)
        print(f"Created backup: {backup_blob_name}")
        
    except Exception as e:
        print(f"No existing historical data found: {e}")
        print("Starting with empty dataframe")
        historical_df = pd.DataFrame()
    
    # Step 2: Get the latest timestamp from historical data
    if len(historical_df) > 0:
        last_timestamp = historical_df.index[-1]
        print(f"Last data point: {last_timestamp}")
        
        # Start from the next period after the last timestamp
        start_date = last_timestamp + timedelta(minutes=15)
    else:
        # If no historical data, start from 60 days ago (Yahoo Finance limit)
        start_date = datetime.now() - timedelta(days=59)
    
    # Step 3: Download new data from Yahoo Finance
    print(f"\nDownloading new data from Yahoo Finance...")
    print(f"Starting from: {start_date}")
    
    ticker = yf.Ticker('NVDA')
    end_date = datetime.now()
    
    # Download new data
    new_data = ticker.history(start=start_date, end=end_date, interval='15m')
    
    if len(new_data) > 0:
        print(f"Downloaded {len(new_data)} new records")
        
        # Format new data to match historical format
        new_data.index.name = 'timestamp'
        new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Step 4: Combine data
        if len(historical_df) > 0:
            # Ensure column names match
            historical_cols = set(historical_df.columns)
            new_cols = set(new_data.columns)
            
            if historical_cols != new_cols:
                print(f"Warning: Column mismatch!")
                print(f"Historical: {historical_cols}")
                print(f"New: {new_cols}")
                
                # Use only common columns
                common_cols = list(historical_cols.intersection(new_cols))
                historical_df = historical_df[common_cols]
                new_data = new_data[common_cols]
            
            # Combine and remove duplicates
            combined_df = pd.concat([historical_df, new_data])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        else:
            combined_df = new_data
        
        print(f"\nCombined dataset: {len(combined_df)} total records")
        print(f"Date range: {combined_df.index[0]} to {combined_df.index[-1]}")
        
        # Step 5: Save locally first
        combined_df.to_csv('data/NVDA_15min_pattern_ready.csv')
        print("Saved updated data locally")
        
        # Step 6: Upload updated file to Azure
        print("Uploading updated data to Azure...")
        blob_client = blob_service.get_blob_client(container=container_name, blob=historical_blob_name)
        
        csv_buffer = io.StringIO()
        combined_df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        blob_client.upload_blob(csv_buffer.getvalue(), overwrite=True)
        print(f"Successfully updated: {historical_blob_name}")
        
        # Summary
        new_records = len(combined_df) - len(historical_df) if len(historical_df) > 0 else len(combined_df)
        print(f"\nSummary: Added {new_records} new records")
        
    else:
        print("No new data available from Yahoo Finance")
        print("This might be because the market is closed or data is already up to date")
        
except Exception as e:
    print(f"Error updating historical data: {e}")
    exit(1)