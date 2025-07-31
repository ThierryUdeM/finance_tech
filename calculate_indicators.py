#!/usr/bin/env python3
"""
Calculate daily indicators for hourly and daily data
This script is part of the hourly/daily workflow and calculates longer-term indicators
"""

import pandas as pd
import numpy as np
import ta
from ta import add_all_ta_features
from ta.utils import dropna
import pyarrow.parquet as pq
import pyarrow as pa
from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime, timedelta

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

def calculate_hourly_indicators(df):
    """Calculate simplified indicators for hourly data"""
    # Sort by datetime
    df = df.sort_values('datetime').copy()
    
    # Ensure datetime is properly handled
    if df['datetime'].dt.tz is not None:
        # If timezone aware, convert to EST
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('America/New_York')
    else:
        # If naive, assume UTC and convert
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    
    # Basic price levels
    df['prior_close'] = df.groupby('ticker')['close'].shift(1)
    
    # Moving averages suitable for hourly
    df['sma_24'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())  # 24h = 1 day
    df['sma_168'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=168, min_periods=1).mean())  # 168h = 1 week
    df['ema_24'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=24, adjust=False).mean())
    
    # RSI
    df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi())
    
    # Volume metrics
    df['avg_volume_24'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())
    df['volume_spike_ratio'] = df['volume'] / df['avg_volume_24']
    
    # Gap metric
    df['gap_pct'] = ((df['open'] - df['prior_close']) / df['prior_close'] * 100).fillna(0)
    
    return df

def calculate_daily_indicators(df):
    """Calculate technical indicators for daily data with appropriate parameters"""
    # Sort by datetime
    df = df.sort_values('datetime').copy()
    
    # Price levels
    df['prior_close'] = df.groupby('ticker')['close'].shift(1)
    
    # Moving averages for daily timeframe
    df['sma_20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    df['sma_100'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=100, min_periods=1).mean())
    df['sma_200'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(window=200, min_periods=1).mean())
    
    df['ema_20'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['ema_50'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=50, adjust=False).mean())
    df['ema_100'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=100, adjust=False).mean())
    df['ema_200'] = df.groupby('ticker')['close'].transform(lambda x: x.ewm(span=200, adjust=False).mean())
    
    # RSI
    df['rsi_14'] = df.groupby('ticker')['close'].transform(lambda x: ta.momentum.RSIIndicator(x, window=14).rsi())
    df['rsi_slope'] = df.groupby('ticker')['rsi_14'].diff()
    
    # MACD
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy()
        macd_indicator = ta.trend.MACD(close=ticker_df['close'], window_slow=26, window_fast=12, window_sign=9)
        df.loc[mask, 'macd'] = macd_indicator.macd()
        df.loc[mask, 'macd_signal'] = macd_indicator.macd_signal()
        df.loc[mask, 'macd_histogram'] = macd_indicator.macd_diff()
    
    # ATR
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy()
        atr_indicator = ta.volatility.AverageTrueRange(
            high=ticker_df['high'], 
            low=ticker_df['low'], 
            close=ticker_df['close'], 
            window=14
        )
        df.loc[mask, 'atr_14'] = atr_indicator.average_true_range()
        df.loc[mask, 'true_range'] = ticker_df['high'] - ticker_df['low']
    
    # Bollinger Bands
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy()
        bb_indicator = ta.volatility.BollingerBands(close=ticker_df['close'], window=20, window_dev=2)
        df.loc[mask, 'bb_upper'] = bb_indicator.bollinger_hband()
        df.loc[mask, 'bb_middle'] = bb_indicator.bollinger_mavg()
        df.loc[mask, 'bb_lower'] = bb_indicator.bollinger_lband()
        df.loc[mask, 'bb_pctb'] = bb_indicator.bollinger_pband()
    
    df['bb_pierce'] = (df['close'] > df['bb_upper']) | (df['close'] < df['bb_lower'])
    
    # Volume metrics
    df['avg_volume_20'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
    df['avg_volume_50'] = df.groupby('ticker')['volume'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())
    df['volume_spike_ratio'] = df['volume'] / df['avg_volume_20']
    
    # OBV
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        ticker_df = df[mask].copy()
        obv_indicator = ta.volume.OnBalanceVolumeIndicator(close=ticker_df['close'], volume=ticker_df['volume'])
        df.loc[mask, 'obv'] = obv_indicator.on_balance_volume()
    
    # Gap metric
    df['gap_pct'] = ((df['open'] - df['prior_close']) / df['prior_close'] * 100).fillna(0)
    
    # Range patterns
    df['range'] = df['high'] - df['low']
    df['nr4'] = df.groupby('ticker')['range'].transform(lambda x: x.rolling(window=4).apply(lambda y: y.iloc[-1] == y.min() if len(y) == 4 else False))
    df['nr7'] = df.groupby('ticker')['range'].transform(lambda x: x.rolling(window=7).apply(lambda y: y.iloc[-1] == y.min() if len(y) == 7 else False))
    
    # 52-week high/low
    df['high_52w'] = df.groupby('ticker')['high'].transform(lambda x: x.rolling(window=252, min_periods=1).max())
    df['low_52w'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(window=252, min_periods=1).min())
    df['pct_from_52w_high'] = ((df['close'] - df['high_52w']) / df['high_52w'] * 100)
    df['pct_from_52w_low'] = ((df['close'] - df['low_52w']) / df['low_52w'] * 100)
    
    # Candlestick patterns (same as intraday)
    df['inside_bar'] = (df.groupby('ticker')['high'].shift(1) >= df['high']) & \
                       (df.groupby('ticker')['low'].shift(1) <= df['low'])
    
    # Hammer pattern
    body = abs(df['close'] - df['open'])
    lower_shadow = np.where(df['close'] >= df['open'], 
                            df['open'] - df['low'], 
                            df['close'] - df['low'])
    upper_shadow = np.where(df['close'] >= df['open'], 
                            df['high'] - df['close'], 
                            df['high'] - df['open'])
    df['hammer'] = (lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)
    
    # Engulfing patterns
    prev_close = df.groupby('ticker')['close'].shift(1)
    prev_open = df.groupby('ticker')['open'].shift(1)
    
    bull_engulf = (df['close'] > df['open']) & \
                  (prev_close < prev_open) & \
                  (df['close'] > prev_open) & \
                  (df['open'] < prev_close)
    
    bear_engulf = (df['close'] < df['open']) & \
                  (prev_close > prev_open) & \
                  (df['close'] < prev_open) & \
                  (df['open'] > prev_close)
    
    df['engulfing'] = bull_engulf | bear_engulf
    
    return df

def load_data_from_azure(container_client, blob_name):
    """Load data from Azure blob storage"""
    try:
        blob_client = container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        df = pq.read_table(pa.BufferReader(blob_data)).to_pandas()
        return df
    except Exception as e:
        print(f"Error loading {blob_name}: {str(e)}")
        return None

def save_data_to_azure(container_client, df, blob_name):
    """Save data to Azure blob storage"""
    try:
        # Ensure timezone is handled properly before saving
        if 'datetime' in df.columns and df['datetime'].dt.tz is not None:
            # Convert timezone-aware to naive (in EST) for storage
            df = df.copy()
            df['datetime'] = df['datetime'].dt.tz_localize(None)
        
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(df), buffer)
        
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(buffer.getvalue().to_pybytes(), overwrite=True)
        print(f"Saved {len(df)} records to {blob_name}")
        return True
    except Exception as e:
        print(f"Error saving to {blob_name}: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("Starting indicator calculation...")
    
    # Get Azure client
    container_client = get_azure_client()
    
    # Process hourly data (intraday indicators)
    print("\n" + "="*60)
    print("Processing hourly data with intraday indicators...")
    
    hourly_data = load_data_from_azure(container_client, "raw_data/raw_data_1h.parquet")
    if hourly_data is not None:
        hourly_with_indicators = calculate_hourly_indicators(hourly_data)
        save_data_to_azure(container_client, hourly_with_indicators, "indicators_azure/data_feed_1h.parquet")
        
        # Also update historic
        historic_hourly = load_data_from_azure(container_client, "raw_data/historic_raw_data_1h.parquet")
        if historic_hourly is not None:
            historic_hourly_indicators = calculate_hourly_indicators(historic_hourly)
            save_data_to_azure(container_client, historic_hourly_indicators, "indicators_azure/historic_data_feed_1h.parquet")
    
    # Process daily data (daily indicators)
    print("\n" + "="*60)
    print("Processing daily data with daily indicators...")
    
    daily_data = load_data_from_azure(container_client, "raw_data/raw_data_1d.parquet")
    if daily_data is not None:
        daily_with_indicators = calculate_daily_indicators(daily_data)
        save_data_to_azure(container_client, daily_with_indicators, "indicators_azure/data_feed_1d.parquet")
        
        # Also update historic
        historic_daily = load_data_from_azure(container_client, "raw_data/historic_raw_data_1d.parquet")
        if historic_daily is not None:
            historic_daily_indicators = calculate_daily_indicators(historic_daily)
            save_data_to_azure(container_client, historic_daily_indicators, "indicators_azure/historic_data_feed_1d.parquet")
    
    print("\nIndicator calculation completed.")

if __name__ == "__main__":
    main()