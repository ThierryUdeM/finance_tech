#!/usr/bin/env python3

import os
import sys
import yfinance as yf
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import pytz

def get_azure_client():
    """Initialize Azure blob storage client"""
    account_name = os.environ.get('STORAGE_ACCOUNT_NAME')
    account_key = os.environ.get('ACCESS_KEY')
    container_name = os.environ.get('CONTAINER_NAME')
    
    if not all([account_name, account_key, container_name]):
        raise ValueError("Azure credentials not found in environment variables")
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    return container_client

def get_tickers():
    """Get tickers from environment variable"""
    tickers_string = os.environ.get('TICKERS')
    if not tickers_string:
        raise ValueError("TICKERS environment variable not found")
    
    tickers = [ticker.strip() for ticker in tickers_string.split(',')]
    return tickers

def fetch_data(tickers, interval, period):
    """Fetch data from yfinance for given tickers and interval"""
    print(f"Fetching {interval} data for {len(tickers)} tickers...")
    
    try:
        # Bulk download for efficiency
        data = yf.download(
            tickers=' '.join(tickers),
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=True
        )
        
        if data.empty:
            print(f"No data retrieved for {interval} interval")
            return pd.DataFrame()
        
        # Process multi-level columns
        combined_data = []
        
        for ticker in tickers:
            try:
                ticker_data = pd.DataFrame()
                
                # Extract data for each metric
                for metric in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if len(tickers) == 1:
                        # Single ticker case - no multi-level columns
                        ticker_data[metric.lower()] = data[metric]
                    else:
                        # Multi-ticker case - multi-level columns
                        ticker_data[metric.lower()] = data[metric][ticker]
                
                # Add ticker column and datetime index
                ticker_data['ticker'] = ticker
                ticker_data['datetime'] = data.index
                ticker_data = ticker_data.reset_index(drop=True)
                
                # Remove rows with all NA values
                ticker_data = ticker_data.dropna(subset=['close'])
                
                if not ticker_data.empty:
                    combined_data.append(ticker_data)
                    print(f"Processed {len(ticker_data)} records for {ticker}")
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            # Ensure datetime is timezone-aware (EST/EDT)
            result['datetime'] = pd.to_datetime(result['datetime']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            return result
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching {interval} data: {str(e)}")
        return pd.DataFrame()

def save_to_azure(container_client, data, interval_name):
    """Save data to Azure blob storage"""
    if data.empty:
        print(f"No data to save for {interval_name}")
        return
    
    try:
        # Define blob names
        current_blob_name = f"raw_data/raw_data_{interval_name}.parquet"
        historic_blob_name = f"raw_data/historic_raw_data_{interval_name}.parquet"
        
        # Save current data
        current_buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(data), current_buffer)
        
        blob_client = container_client.get_blob_client(current_blob_name)
        blob_client.upload_blob(current_buffer.getvalue().to_pybytes(), overwrite=True)
        print(f"Uploaded current {interval_name} data to: {current_blob_name}")
        
        # Handle historic data
        try:
            # Download existing historic data
            blob_client_historic = container_client.get_blob_client(historic_blob_name)
            existing_data = blob_client_historic.download_blob().readall()
            existing_df = pq.read_table(pa.BufferReader(existing_data)).to_pandas()
            
            # Get today's date
            today_date = pd.Timestamp.now(tz='America/New_York').date()
            
            # Remove today's data from historic
            existing_df['date'] = pd.to_datetime(existing_df['datetime']).dt.date
            existing_df_clean = existing_df[existing_df['date'] != today_date].drop(columns=['date'])
            
            # Combine with new data
            updated_df = pd.concat([existing_df_clean, data], ignore_index=True)
            
        except Exception:
            # No existing historic data
            print(f"No existing historic data found for {interval_name}, creating new file")
            updated_df = data
        
        # Upload historic data
        historic_buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(updated_df), historic_buffer)
        
        blob_client_historic = container_client.get_blob_client(historic_blob_name)
        blob_client_historic.upload_blob(historic_buffer.getvalue().to_pybytes(), overwrite=True)
        print(f"Updated historic {interval_name} data with {len(data)} new records")
        
        # Print summary
        print(f"Total records: {len(data)}")
        print(f"Date range: {data['datetime'].min()} to {data['datetime'].max()}")
        print(f"Unique tickers: {data['ticker'].nunique()}")
        
    except Exception as e:
        print(f"Error uploading to Azure: {str(e)}")

def main():
    """Main execution function"""
    print("Starting hourly and daily data fetch...")
    
    # Get configuration
    tickers = get_tickers()
    container_client = get_azure_client()
    
    # Fetch hourly data (last 60 days)
    print("\n" + "="*60)
    print("Fetching hourly data...")
    hourly_data = fetch_data(tickers, '1h', '60d')
    save_to_azure(container_client, hourly_data, '1h')
    
    # Fetch daily data (last 2 years)
    print("\n" + "="*60)
    print("Fetching daily data...")
    daily_data = fetch_data(tickers, '1d', '2y')
    save_to_azure(container_client, daily_data, '1d')
    
    print("\nData fetching completed for hourly and daily intervals.")

if __name__ == "__main__":
    main()