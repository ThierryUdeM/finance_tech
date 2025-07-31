#!/usr/bin/env python3
"""
Advanced Pattern Detection for Daily and Hourly Data
Detects trend continuation, breakout, mean reversion, and reversal patterns
Sends email alerts when hourly cues are triggered
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from azure.storage.blob import BlobServiceClient
import os
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

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

def send_email_alert(subject, body, ticker, pattern_type):
    """Send email alert using Gmail SMTP"""
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_pwd = os.environ.get('GMAIL_APP_PWD')
    alert_to = os.environ.get('ALERT_TO')
    
    if not all([gmail_user, gmail_pwd, alert_to]):
        print("Email credentials not found in environment variables")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = alert_to
        msg['Subject'] = f"🚨 Trading Alert: {ticker} - {subject}"
        
        # Create HTML body
        html_body = f"""
        <html>
            <body>
                <h2>Trading Pattern Alert</h2>
                <h3>Pattern: {pattern_type}</h3>
                <h4>Ticker: {ticker}</h4>
                <p>{body}</p>
                <hr>
                <p><small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST</small></p>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.send_message(msg)
        server.quit()
        
        print(f"Email alert sent for {ticker} - {pattern_type}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False

def check_trend_continuation_patterns(daily_df, hourly_df):
    """Check for trend continuation patterns"""
    alerts = []
    
    for ticker in daily_df['ticker'].unique():
        daily_ticker = daily_df[daily_df['ticker'] == ticker].sort_values('datetime')
        hourly_ticker = hourly_df[hourly_df['ticker'] == ticker].sort_values('datetime')
        
        if len(daily_ticker) < 50 or len(hourly_ticker) < 24:
            continue
            
        # Get latest daily data
        latest_daily = daily_ticker.iloc[-1]
        prev_daily = daily_ticker.iloc[-2] if len(daily_ticker) > 1 else None
        
        # 1. Golden/Silver Cross Pattern
        if (latest_daily['ema_20'] > latest_daily['ema_50'] and 
            prev_daily is not None and prev_daily['ema_20'] <= prev_daily['ema_50'] and
            latest_daily['close'] >= latest_daily['sma_50'] and 
            latest_daily['rsi_14'] > 55):
            
            # Check hourly cue
            latest_hourly = hourly_ticker.iloc[-1]
            if (latest_hourly['close'] > latest_hourly['ema_24'] and
                latest_hourly['volume_spike_ratio'] >= 1.2):
                
                alert = {
                    'ticker': ticker,
                    'pattern': 'Golden Cross',
                    'daily_trigger': 'EMA20 crossed above EMA50, RSI > 55',
                    'hourly_cue': 'Price holding above EMA24 with volume spike',
                    'action': 'BUY',
                    'risk_level': 'LOW'
                }
                alerts.append(alert)
        
        # 2. 20-EMA Bounce Pattern
        if (latest_daily['close'] > latest_daily['ema_20'] > latest_daily['ema_50'] and
            latest_daily['low'] <= latest_daily['ema_20'] and
            latest_daily['close'] > latest_daily['open']):  # Green candle
            
            # Check hourly cue
            latest_hourly = hourly_ticker.iloc[-1]
            if (latest_hourly.get('hammer', False) or latest_hourly.get('engulfing', False)):
                # Check OBV uptick
                if len(hourly_ticker) > 1:
                    obv_change = latest_hourly['obv'] - hourly_ticker.iloc[-2]['obv']
                    if obv_change > 0:
                        alert = {
                            'ticker': ticker,
                            'pattern': '20-EMA Bounce',
                            'daily_trigger': 'Price bounced off EMA20 in uptrend',
                            'hourly_cue': 'Bullish pattern with OBV uptick',
                            'action': 'BUY',
                            'risk_level': 'LOW'
                        }
                        alerts.append(alert)
    
    return alerts

def check_breakout_patterns(daily_df, hourly_df):
    """Check for breakout/expansion patterns"""
    alerts = []
    
    for ticker in daily_df['ticker'].unique():
        daily_ticker = daily_df[daily_df['ticker'] == ticker].sort_values('datetime')
        hourly_ticker = hourly_df[hourly_df['ticker'] == ticker].sort_values('datetime')
        
        if len(daily_ticker) < 252:  # Need 52 weeks of data
            continue
            
        latest_daily = daily_ticker.iloc[-1]
        
        # 1. NR7 + Inside Bar Combo
        if (latest_daily.get('nr7', False) and 
            latest_daily.get('inside_bar', False) and 
            latest_daily.get('bb_pctb', 1) < 0.2):
            
            # Check hourly cue
            latest_hourly = hourly_ticker.iloc[-1]
            if latest_hourly['volume_spike_ratio'] >= 1.5:
                alert = {
                    'ticker': ticker,
                    'pattern': 'NR7 + Inside Bar Breakout',
                    'daily_trigger': 'NR7 with inside bar, low BB position',
                    'hourly_cue': 'Volume surge on range break',
                    'action': 'BUY on breakout',
                    'risk_level': 'LOW'
                }
                alerts.append(alert)
        
        # 2. 52-Week High Break
        if (latest_daily['close'] >= latest_daily['high_52w'] and
            latest_daily['volume_spike_ratio'] >= 1.5):
            
            # Check hourly cue (next day's first bullish pattern)
            latest_hourly = hourly_ticker.iloc[-1]
            if latest_hourly['close'] > latest_daily['close']:  # Trading above breakout
                alert = {
                    'ticker': ticker,
                    'pattern': '52-Week High Breakout',
                    'daily_trigger': 'New 52w high with volume surge',
                    'hourly_cue': 'Holding above breakout level',
                    'action': 'BUY',
                    'risk_level': 'MEDIUM'
                }
                alerts.append(alert)
        
        # 3. Gap-and-Go Swing
        if (latest_daily['gap_pct'] > 2 and
            latest_daily['close'] > latest_daily['open'] and
            latest_daily['rsi_14'] > 60):
            
            # Check hourly cue
            gap_midpoint = (latest_daily['open'] + daily_ticker.iloc[-2]['close']) / 2
            latest_hourly = hourly_ticker.iloc[-1]
            if latest_hourly['close'] > gap_midpoint:
                alert = {
                    'ticker': ticker,
                    'pattern': 'Gap-and-Go',
                    'daily_trigger': f'Gap up {latest_daily["gap_pct"]:.1f}%, RSI > 60',
                    'hourly_cue': 'Holding above gap midpoint',
                    'action': 'BUY',
                    'risk_level': 'MEDIUM'
                }
                alerts.append(alert)
    
    return alerts

def check_mean_reversion_patterns(daily_df, hourly_df):
    """Check for mean reversion/exhaustion patterns"""
    alerts = []
    
    for ticker in daily_df['ticker'].unique():
        daily_ticker = daily_df[daily_df['ticker'] == ticker].sort_values('datetime')
        hourly_ticker = hourly_df[hourly_df['ticker'] == ticker].sort_values('datetime')
        
        if len(daily_ticker) < 20:
            continue
            
        latest_daily = daily_ticker.iloc[-1]
        
        # 1. Bollinger Band Pierce + Oversold RSI
        if (latest_daily['close'] < latest_daily['bb_lower'] and
            latest_daily['rsi_14'] <= 30 and
            (latest_daily.get('hammer', False) or latest_daily.get('engulfing', False))):
            
            # Check hourly cue
            latest_hourly = hourly_ticker.iloc[-1]
            if (latest_hourly.get('engulfing', False) or 
                latest_hourly.get('rsi_14', 0) > 40):
                
                alert = {
                    'ticker': ticker,
                    'pattern': 'BB Pierce Reversal',
                    'daily_trigger': 'Below BB lower, RSI < 30, reversal candle',
                    'hourly_cue': 'Bullish reversal signal',
                    'action': 'BUY',
                    'risk_level': 'MEDIUM',
                    'target': f"BB Middle: ${latest_daily['bb_middle']:.2f}"
                }
                alerts.append(alert)
        
        # 2. Over-extension Fade
        if latest_daily['atr_14'] > 0:
            extension = (latest_daily['close'] - latest_daily['ema_20']) / latest_daily['atr_14']
            if extension >= 2 and latest_daily['rsi_14'] > 75:
                
                # Check hourly cue
                if len(hourly_ticker) > 1:
                    latest_hourly = hourly_ticker.iloc[-1]
                    prev_hourly = hourly_ticker.iloc[-2]
                    
                    if (latest_hourly['high'] < prev_hourly['high'] and  # Lower high
                        latest_hourly.get('engulfing', False) and
                        latest_hourly['volume'] < prev_hourly['volume']):  # Volume drying
                        
                        alert = {
                            'ticker': ticker,
                            'pattern': 'Over-extension Fade',
                            'daily_trigger': f'{extension:.1f} ATR above EMA20, RSI > 75',
                            'hourly_cue': 'Lower high with bearish engulfing',
                            'action': 'SHORT/SELL',
                            'risk_level': 'HIGH'
                        }
                        alerts.append(alert)
    
    return alerts

def check_reversal_patterns(daily_df, hourly_df):
    """Check for reversal patterns at key structure levels"""
    alerts = []
    
    for ticker in daily_df['ticker'].unique():
        daily_ticker = daily_df[daily_df['ticker'] == ticker].sort_values('datetime')
        hourly_ticker = hourly_df[hourly_df['ticker'] == ticker].sort_values('datetime')
        
        if len(daily_ticker) < 200:
            continue
            
        latest_daily = daily_ticker.iloc[-1]
        
        # 1. Hammer at Major MA
        if latest_daily.get('hammer', False):
            # Check if at 50-SMA or 200-SMA
            at_50ma = abs(latest_daily['low'] - latest_daily['sma_50']) / latest_daily['sma_50'] < 0.01
            at_200ma = abs(latest_daily['low'] - latest_daily['sma_200']) / latest_daily['sma_200'] < 0.01
            
            if at_50ma or at_200ma:
                ma_level = 'SMA50' if at_50ma else 'SMA200'
                
                # Check hourly cue
                latest_hourly = hourly_ticker.iloc[-1]
                if (latest_hourly.get('inside_bar', False) and 
                    latest_hourly['close'] > latest_daily['high']):
                    
                    alert = {
                        'ticker': ticker,
                        'pattern': f'Hammer at {ma_level}',
                        'daily_trigger': f'Hammer rejection at {ma_level}',
                        'hourly_cue': 'Inside bar break above hammer high',
                        'action': 'BUY',
                        'risk_level': 'LOW'
                    }
                    alerts.append(alert)
        
        # 2. Engulfing after drift
        if latest_daily.get('engulfing', False):
            # Check if after multi-day drift (low volatility)
            recent_ranges = daily_ticker.tail(5)['range'].mean()
            if recent_ranges < daily_ticker.tail(20)['range'].mean() * 0.7:  # Volatility contraction
                
                # Check hourly cue and OBV divergence
                latest_hourly = hourly_ticker.iloc[-1]
                if len(hourly_ticker) > 5:
                    obv_slope = np.polyfit(range(5), hourly_ticker.tail(5)['obv'].values, 1)[0]
                    price_slope = np.polyfit(range(5), hourly_ticker.tail(5)['close'].values, 1)[0]
                    
                    # Bullish divergence: OBV rising while price falling
                    if obv_slope > 0 and price_slope < 0 and latest_daily['close'] > latest_daily['open']:
                        alert = {
                            'ticker': ticker,
                            'pattern': 'Bullish Engulfing Reversal',
                            'daily_trigger': 'Engulfing after low volatility drift',
                            'hourly_cue': 'OBV divergence confirmed',
                            'action': 'BUY',
                            'risk_level': 'MEDIUM'
                        }
                        alerts.append(alert)
    
    return alerts

def check_volume_confirmation(alerts, hourly_df):
    """Add volume confirmation to existing alerts"""
    confirmed_alerts = []
    
    for alert in alerts:
        ticker = alert['ticker']
        hourly_ticker = hourly_df[hourly_df['ticker'] == ticker].sort_values('datetime')
        
        if len(hourly_ticker) > 0:
            latest_hourly = hourly_ticker.iloc[-1]
            
            # Check volume spike and OBV slope
            volume_confirmed = latest_hourly.get('volume_spike_ratio', 0) >= 1.3
            
            if len(hourly_ticker) > 2:
                obv_slope = hourly_ticker.tail(3)['obv'].diff().mean()
                obv_confirmed = (obv_slope > 0 and alert['action'].startswith('BUY')) or \
                               (obv_slope < 0 and alert['action'].startswith('SHORT'))
            else:
                obv_confirmed = True  # Default to confirmed if not enough data
            
            if volume_confirmed and obv_confirmed:
                alert['volume_confirmed'] = True
                alert['confidence'] = 'HIGH'
            else:
                alert['volume_confirmed'] = False
                alert['confidence'] = 'MEDIUM'
            
            confirmed_alerts.append(alert)
    
    return confirmed_alerts

def save_patterns_to_azure(container_client, patterns_df, blob_name):
    """Save detected patterns to Azure"""
    try:
        buffer = pa.BufferOutputStream()
        pq.write_table(pa.Table.from_pandas(patterns_df), buffer)
        
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(buffer.getvalue().to_pybytes(), overwrite=True)
        print(f"Saved {len(patterns_df)} patterns to {blob_name}")
        return True
    except Exception as e:
        print(f"Error saving patterns: {str(e)}")
        return False

def main():
    """Main execution function"""
    print("Starting advanced pattern detection...")
    
    # Get Azure client
    container_client = get_azure_client()
    
    # Load daily and hourly data with indicators
    daily_df = load_data_from_azure(container_client, "indicators_azure/data_feed_1d.parquet")
    hourly_df = load_data_from_azure(container_client, "indicators_azure/data_feed_1h.parquet")
    
    if daily_df is None or hourly_df is None:
        print("Error: Could not load indicator data")
        return
    
    print(f"Loaded {len(daily_df)} daily records and {len(hourly_df)} hourly records")
    
    # Get latest data only (last 2 days for daily, last 48 hours for hourly)
    latest_date = daily_df['datetime'].max()
    cutoff_daily = latest_date - timedelta(days=2)
    cutoff_hourly = hourly_df['datetime'].max() - timedelta(hours=48)
    
    recent_daily = daily_df[daily_df['datetime'] >= cutoff_daily]
    recent_hourly = hourly_df[hourly_df['datetime'] >= cutoff_hourly]
    
    # Check all pattern types
    all_alerts = []
    
    print("\nChecking trend continuation patterns...")
    trend_alerts = check_trend_continuation_patterns(recent_daily, recent_hourly)
    all_alerts.extend(trend_alerts)
    
    print("Checking breakout patterns...")
    breakout_alerts = check_breakout_patterns(recent_daily, recent_hourly)
    all_alerts.extend(breakout_alerts)
    
    print("Checking mean reversion patterns...")
    reversion_alerts = check_mean_reversion_patterns(recent_daily, recent_hourly)
    all_alerts.extend(reversion_alerts)
    
    print("Checking reversal patterns...")
    reversal_alerts = check_reversal_patterns(recent_daily, recent_hourly)
    all_alerts.extend(reversal_alerts)
    
    # Add volume confirmation
    print("Adding volume confirmation...")
    confirmed_alerts = check_volume_confirmation(all_alerts, recent_hourly)
    
    print(f"\nTotal patterns detected: {len(confirmed_alerts)}")
    
    # Send email alerts for high-confidence patterns
    if confirmed_alerts:
        high_confidence = [a for a in confirmed_alerts if a.get('confidence') == 'HIGH']
        print(f"High confidence patterns: {len(high_confidence)}")
        
        for alert in high_confidence:
            body = f"""
            Pattern Type: {alert['pattern']}
            Daily Trigger: {alert['daily_trigger']}
            Hourly Cue: {alert['hourly_cue']}
            Action: {alert['action']}
            Risk Level: {alert['risk_level']}
            Volume Confirmed: {alert['volume_confirmed']}
            Confidence: {alert['confidence']}
            """
            
            if 'target' in alert:
                body += f"\nTarget: {alert['target']}"
            
            send_email_alert(
                subject=f"{alert['pattern']} Alert",
                body=body,
                ticker=alert['ticker'],
                pattern_type=alert['pattern']
            )
    
    # Save all patterns to Azure
    if confirmed_alerts:
        patterns_df = pd.DataFrame(confirmed_alerts)
        patterns_df['timestamp'] = datetime.now()
        save_patterns_to_azure(
            container_client, 
            patterns_df, 
            "patterns/advanced_patterns_latest.parquet"
        )
        
        # Also save timestamped version
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_patterns_to_azure(
            container_client,
            patterns_df,
            f"patterns/advanced_patterns_{timestamp_str}.parquet"
        )
    
    print("\nPattern detection completed.")

if __name__ == "__main__":
    main()