#!/usr/bin/env python3
"""
Standalone signal detector for GitHub Actions
Fetches data from yfinance and detects trading signals
"""

import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Technical indicators
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_vwap(df):
    return (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

def calculate_atr(high, low, close, period=14):
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr

# Strategy detection functions
def detect_orb(df, current_idx):
    """Opening Range Breakout"""
    if current_idx < 30:  # Need 30 minutes of data
        return None
    
    # Get first 30 minutes
    opening_range = df.iloc[:30]
    or_high = opening_range['High'].max()
    or_low = opening_range['Low'].min()
    
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    
    # Breakout above OR high
    if prev['Close'] <= or_high and current['Close'] > or_high:
        if current['Volume'] > df['Volume'].rolling(20).mean().iloc[current_idx] * 1.5:
            return {
                'strategy': 'ORB',
                'direction': 'LONG',
                'entry': current['Close'],
                'stop': or_low,
                'target': current['Close'] + 2 * (current['Close'] - or_low),
                'confidence': 75
            }
    
    return None

def detect_vwap_bounce(df, current_idx):
    """VWAP Bounce/Rejection"""
    if current_idx < 20:
        return None
    
    current = df.iloc[current_idx]
    vwap = df['VWAP'].iloc[current_idx]
    atr = df['ATR'].iloc[current_idx]
    
    # Price near VWAP (within 0.2 ATR)
    if abs(current['Close'] - vwap) < 0.2 * atr:
        # Check for bounce
        if current['Low'] <= vwap and current['Close'] > vwap:
            # Bullish bounce
            if current['RSI'] > 40 and current['RSI'] < 60:
                return {
                    'strategy': 'VWAP_Bounce',
                    'direction': 'LONG',
                    'entry': current['Close'],
                    'stop': vwap - 0.5 * atr,
                    'target': current['Close'] + 1.5 * atr,
                    'confidence': 80
                }
    
    return None

def detect_ema_pullback(df, current_idx):
    """EMA 9/20 Pullback"""
    if current_idx < 20:
        return None
    
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    
    # Uptrend: EMA9 > EMA20 > VWAP
    if current['EMA9'] > current['EMA20'] and current['EMA20'] > current['VWAP']:
        # Pullback to EMA20 and bounce
        if prev['Low'] <= current['EMA20'] and current['Close'] > current['EMA9']:
            # Volume confirmation
            if current['Volume'] > df['Volume'].rolling(20).mean().iloc[current_idx] * 1.3:
                return {
                    'strategy': 'EMA_Pullback',
                    'direction': 'LONG',
                    'entry': current['Close'],
                    'stop': current['EMA20'] - 0.3 * current['ATR'],
                    'target': current['Close'] + 1.5 * current['ATR'],
                    'confidence': 82
                }
    
    return None

def detect_support_bounce(df, current_idx):
    """Support Bounce Pattern"""
    if current_idx < 10:
        return None
    
    current = df.iloc[current_idx]
    recent_bars = df.iloc[max(0, current_idx-10):current_idx]
    
    # Find recent low
    recent_low = recent_bars['Low'].min()
    
    # Check if we bounced from support
    if current['Low'] <= recent_low * 1.001 and current['Close'] > current['Low'] + 0.6 * (current['High'] - current['Low']):
        # RSI oversold bounce
        if current['RSI'] < 40:
            return {
                'strategy': 'Support_Bounce',
                'direction': 'LONG',
                'entry': current['Close'],
                'stop': recent_low - 0.2 * current['ATR'],
                'target': current['Close'] + 2 * current['ATR'],
                'confidence': 78
            }
    
    return None

def fetch_and_analyze(ticker, use_last_trading_day=False):
    """Fetch data and run signal detection"""
    try:
        # Download recent data (5 days of 1-minute bars)
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="1m")
        
        # For Canadian stocks, also try 2m interval if 1m is empty
        if df.empty and ticker.endswith('.TO'):
            df = stock.history(period="5d", interval="2m")
        
        if df.empty:
            return None
        
        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        df['VWAP'] = calculate_vwap(df)
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['EMA9'] = df['Close'].ewm(span=9).mean()
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        
        # Get target day's data
        if use_last_trading_day:
            # Get the most recent trading day
            available_dates = df.index.date
            unique_dates = sorted(set(available_dates), reverse=True)
            if len(unique_dates) > 0:
                target_date = unique_dates[0]  # Most recent date
                df_today = df[df.index.date == target_date]
                print(f"  Using last trading day data: {target_date}")
            else:
                return None
        else:
            # Get today's data only
            today = datetime.now().date()
            df_today = df[df.index.date == today]
        
        if len(df_today) < 30:  # Need at least 30 minutes
            if not use_last_trading_day:
                print(f"  Insufficient data for today, only {len(df_today)} bars")
            return None
        
        # Run strategies on latest bar
        current_idx = len(df_today) - 1
        
        # Try each strategy
        strategies = [
            detect_orb,
            detect_vwap_bounce,
            detect_ema_pullback,
            detect_support_bounce
        ]
        
        for strategy_func in strategies:
            signal = strategy_func(df_today, current_idx)
            if signal and signal['confidence'] >= 78:  # 78% threshold
                signal['ticker'] = ticker
                signal['time'] = df_today.index[current_idx].strftime('%Y-%m-%d %H:%M:%S')
                signal['current_price'] = df_today.iloc[current_idx]['Close']
                return signal
        
        return None
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

def send_email_alert(signals):
    """Send email alert for detected signals"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Email configuration from environment/secrets
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_pwd = os.environ.get('GMAIL_APP_PWD')
    alert_to = os.environ.get('ALERT_TO')
    
    if not all([gmail_user, gmail_pwd, alert_to]):
        print("Email configuration missing")
        return False
    
    # Create email content
    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🚨 Trading Alert: {len(signals)} New Signal(s)"
    msg['From'] = gmail_user
    msg['To'] = alert_to
    
    # HTML content
    html_parts = ["<h2>New Trading Signals Detected</h2>"]
    
    for signal in signals:
        rr_ratio = abs((signal['target'] - signal['entry']) / (signal['entry'] - signal['stop']))
        
        html_parts.append(f"""
        <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">
            <h3>{signal['ticker']} - {signal['strategy']}</h3>
            <table style="width: 100%;">
                <tr><td><b>Time:</b></td><td>{signal['time']}</td></tr>
                <tr><td><b>Direction:</b></td><td style="color: green;"><b>LONG</b></td></tr>
                <tr><td><b>Entry:</b></td><td>${signal['entry']:.2f}</td></tr>
                <tr><td><b>Stop:</b></td><td>${signal['stop']:.2f}</td></tr>
                <tr><td><b>Target:</b></td><td>${signal['target']:.2f}</td></tr>
                <tr><td><b>R:R:</b></td><td>{rr_ratio:.1f}:1</td></tr>
                <tr><td><b>Confidence:</b></td><td style="background-color: #e6ffe6;"><b>{signal['confidence']}%</b></td></tr>
            </table>
        </div>
        """)
    
    # Check if market is closed
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    market_open = (
        now_et.weekday() < 5 and
        ((now_et.hour == 9 and now_et.minute >= 30) or 
         (now_et.hour > 9 and now_et.hour < 16))
    )
    
    data_note = ""
    if not market_open:
        data_note = "<br><b>Note:</b> Market is closed. Signals based on last trading day's closing data."
    
    html_parts.append(f"""
    <p style="margin-top: 20px; color: #666;">
    <i>Automated signal detection via GitHub Actions<br>
    Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC{data_note}</i>
    </p>
    """)
    
    html = ''.join(html_parts)
    msg.attach(MIMEText(html, 'html'))
    
    # Send email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_user, gmail_pwd)
            server.send_message(msg)
        print(f"Email sent successfully with {len(signals)} signals")
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def main():
    """Main function"""
    # Get tickers from environment or use defaults
    tickers_env = os.environ.get('TICKERS', '')
    if tickers_env:
        tickers = [t.strip() for t in tickers_env.split(',')]
    else:
        # Default tickers
        tickers = ['NVDA', 'MSFT', 'TSLA', 'AAPL', 'GOOG', 'META', 'AMZN']
    
    print(f"Scanning {len(tickers)} tickers for signals...")
    
    # Check market hours (US Eastern Time)
    from datetime import timezone
    import pytz
    
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    
    # Check if market is open
    market_open = (
        now_et.weekday() < 5 and  # Monday-Friday
        ((now_et.hour == 9 and now_et.minute >= 30) or 
         (now_et.hour > 9 and now_et.hour < 16))
    )
    
    # Check if running in test mode
    test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
    
    if not market_open:
        if test_mode:
            print(f"Market is closed (ET: {now_et.strftime('%H:%M')}). Running in TEST MODE using last trading day data.")
            use_last_trading_day = True
        else:
            print(f"Market is closed (ET: {now_et.strftime('%H:%M')}). Using last trading day data for analysis.")
            use_last_trading_day = True
    else:
        print(f"Market is OPEN (ET: {now_et.strftime('%H:%M')}). Using live data.")
        use_last_trading_day = False
    
    # Analyze each ticker
    detected_signals = []
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        signal = fetch_and_analyze(ticker, use_last_trading_day)
        if signal:
            detected_signals.append(signal)
            print(f"  ✓ Signal detected: {signal['strategy']} @ ${signal['entry']:.2f}")
    
    # Send email if signals found
    if detected_signals:
        print(f"\nFound {len(detected_signals)} signals!")
        if not market_open:
            print("Note: These signals are from the last trading day's closing data.")
        send_email_alert(detected_signals)
    else:
        print("\nNo signals detected.")
    
    # Output signals as JSON for workflow artifact
    with open('signals.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'market_status': 'open' if market_open else 'closed',
            'data_source': 'live' if market_open else 'last_trading_day',
            'signals': detected_signals,
            'tickers_scanned': len(tickers)
        }, f, indent=2)

if __name__ == "__main__":
    main()