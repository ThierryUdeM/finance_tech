#!/usr/bin/env python3
"""
Flexible Signal Detector with Dynamic Parameters
Adapts strategy parameters based on market conditions, volatility, and time of day
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength"""
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr = true_range = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx

# Helper functions for flexible parameters
def get_time_adjusted_vol_mult(current_time, base_mult):
    """Adjust volume multiplier based on time of day"""
    hour = current_time.hour + current_time.minute / 60
    
    # Market open at 9:30 AM
    market_open = 9.5
    market_hours = 6.5
    
    if hour < market_open:
        return base_mult
    
    time_elapsed = min(hour - market_open, market_hours)
    decay_factor = np.sqrt(time_elapsed / market_hours)
    
    # Reduce volume requirement as day progresses (up to 30% reduction)
    adjusted_mult = base_mult * (1 - 0.3 * decay_factor)
    return max(adjusted_mult, 1.1)  # Never go below 1.1x

def get_atr_based_threshold(atr, price, base_pct, atr_mult):
    """Use the larger of percentage-based or ATR-based threshold"""
    pct_threshold = price * base_pct
    atr_threshold = atr * atr_mult
    return max(pct_threshold, atr_threshold)

def get_market_regime(df, lookback=20):
    """Detect current market regime"""
    if len(df) < lookback:
        return "unknown"
    
    # Calculate ADX
    try:
        adx = calculate_adx(df['High'], df['Low'], df['Close'])
        adx_val = adx.iloc[-1]
    except:
        adx_val = None
    
    # Calculate ATR/price ratio
    atr = df['ATR'].iloc[-1]
    price = df['Close'].iloc[-1]
    atr_ratio = atr / price
    
    # Determine regime
    if adx_val is not None:
        if adx_val > 30:
            return "trending"
        if adx_val < 20:
            return "ranging"
    
    # Fallback to ATR ratio
    if atr_ratio > 0.02:
        return "volatile"
    if atr_ratio < 0.01:
        return "quiet"
    
    return "normal"

def should_skip_strategy(strategy_type, current_hour):
    """Determine if strategy should be skipped based on time"""
    # First 15 minutes - skip reversals
    if current_hour < 9.75 and strategy_type == "reversal":
        return True
    
    # Lunch hour - be more selective
    if 11.5 <= current_hour <= 13.0:
        if strategy_type in ["breakout", "momentum"]:
            return np.random.random() > 0.7  # Skip 30% randomly
    
    # Last 30 minutes - focus on momentum
    if current_hour >= 15.5 and strategy_type == "mean_reversion":
        return True
    
    return False

# Flexible strategy detection functions
def detect_orb_flexible(df, current_idx, market_regime="normal"):
    """Opening Range Breakout with flexible parameters"""
    # Base parameters
    base_or_length = 30  # minutes
    base_vol_mult = 1.5
    base_or_width = 0.005
    
    # Adjust based on market regime
    if market_regime == "volatile":
        or_length = 45  # Longer OR in volatile markets
        vol_mult = 1.3  # Lower volume requirement
        min_or_width = 0.008  # Wider range required
    elif market_regime == "quiet":
        or_length = 20  # Shorter OR in quiet markets
        vol_mult = 2.0  # Higher volume requirement
        min_or_width = 0.003  # Tighter range acceptable
    else:
        or_length = base_or_length
        vol_mult = base_vol_mult
        min_or_width = base_or_width
    
    if current_idx < or_length:
        return None
    
    # Get opening range
    opening_range = df.iloc[:or_length]
    or_high = opening_range['High'].max()
    or_low = opening_range['Low'].min()
    or_width = (or_high - or_low) / or_low
    
    # Check minimum width
    if or_width < min_or_width:
        return None
    
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    
    # Time-adjusted volume multiplier
    current_time = current.name
    adjusted_vol_mult = get_time_adjusted_vol_mult(current_time, vol_mult)
    
    # ATR-based breakout threshold
    atr = current['ATR']
    breakout_buffer = get_atr_based_threshold(atr, or_high, 0.0003, 0.1)
    
    # Check for breakout
    if prev['Close'] <= or_high and current['Close'] > or_high + breakout_buffer:
        vol_avg = df['Volume'].rolling(20).mean().iloc[current_idx]
        if current['Volume'] > vol_avg * adjusted_vol_mult:
            # Dynamic stop based on ATR
            stop_distance = max(or_high - or_low, atr * 0.5)
            
            return {
                'strategy': 'ORB_Flexible',
                'direction': 'LONG',
                'entry': current['Close'],
                'stop': current['Close'] - stop_distance,
                'target': current['Close'] + stop_distance * 2,
                'confidence': calculate_flexible_confidence(current, df, current_idx, 'breakout'),
                'regime': market_regime
            }
    
    return None

def detect_vwap_bounce_flexible(df, current_idx, market_regime="normal"):
    """VWAP Bounce with flexible parameters"""
    if current_idx < 20:
        return None
    
    current = df.iloc[current_idx]
    vwap = df['VWAP'].iloc[current_idx]
    atr = df['ATR'].iloc[current_idx]
    
    # Flexible distance from VWAP based on regime
    if market_regime == "volatile":
        vwap_distance = 0.3 * atr  # Allow more distance in volatile markets
        rsi_range = (35, 65)  # Wider RSI range
    else:
        vwap_distance = 0.2 * atr
        rsi_range = (40, 60)
    
    # Check if price is near VWAP
    if abs(current['Close'] - vwap) < vwap_distance:
        # Check for bounce
        if current['Low'] <= vwap and current['Close'] > vwap:
            # Check RSI
            if rsi_range[0] < current['RSI'] < rsi_range[1]:
                # Dynamic targets based on ATR
                stop_distance = atr * 0.5
                target_distance = atr * 1.5
                
                # Adjust for time of day
                current_hour = current.name.hour + current.name.minute / 60
                if current_hour > 14.5:  # Late day - smaller targets
                    target_distance *= 0.7
                
                return {
                    'strategy': 'VWAP_Bounce_Flexible',
                    'direction': 'LONG',
                    'entry': current['Close'],
                    'stop': vwap - stop_distance,
                    'target': current['Close'] + target_distance,
                    'confidence': calculate_flexible_confidence(current, df, current_idx, 'mean_reversion'),
                    'regime': market_regime
                }
    
    return None

def detect_ema_pullback_flexible(df, current_idx, market_regime="normal"):
    """EMA Pullback with flexible parameters"""
    if current_idx < 20:
        return None
    
    current = df.iloc[current_idx]
    prev = df.iloc[current_idx - 1]
    
    # Check trend
    if not (current['EMA_9'] > current['EMA_21'] and current['EMA_21'] > current['VWAP']):
        return None
    
    # Flexible pullback depth based on regime
    if market_regime == "trending":
        max_pullback = current['ATR'] * 0.7  # Allow deeper pullbacks in trends
        vol_mult = 1.2  # Lower volume requirement
    else:
        max_pullback = current['ATR'] * 0.5
        vol_mult = 1.3
    
    # Check for pullback and bounce
    if prev['Low'] <= current['EMA_21'] and current['Close'] > current['EMA_9']:
        pullback_depth = current['EMA_21'] - prev['Low']
        
        if pullback_depth <= max_pullback:
            # Volume confirmation with time adjustment
            current_time = current.name
            adjusted_vol_mult = get_time_adjusted_vol_mult(current_time, vol_mult)
            vol_avg = df['Volume'].rolling(20).mean().iloc[current_idx]
            
            if current['Volume'] > vol_avg * adjusted_vol_mult:
                return {
                    'strategy': 'EMA_Pullback_Flexible',
                    'direction': 'LONG',
                    'entry': current['Close'],
                    'stop': current['EMA_21'] - current['ATR'] * 0.3,
                    'target': current['Close'] + current['ATR'] * 1.5,
                    'confidence': calculate_flexible_confidence(current, df, current_idx, 'momentum'),
                    'regime': market_regime
                }
    
    return None

def detect_support_bounce_flexible(df, current_idx, market_regime="normal"):
    """Support Bounce with flexible parameters"""
    if current_idx < 10:
        return None
    
    current = df.iloc[current_idx]
    
    # Dynamic lookback based on regime
    if market_regime == "volatile":
        lookback = 15  # Look further back in volatile markets
        touch_tolerance = 0.3 * current['ATR']
    else:
        lookback = 10
        touch_tolerance = 0.2 * current['ATR']
    
    recent_bars = df.iloc[max(0, current_idx-lookback):current_idx]
    recent_low = recent_bars['Low'].min()
    
    # Check if we're at support
    if abs(current['Low'] - recent_low) < touch_tolerance:
        # Check for bounce
        bounce_strength = (current['Close'] - current['Low']) / (current['High'] - current['Low'])
        
        if bounce_strength > 0.6:  # Strong close
            # Flexible RSI based on regime
            rsi_threshold = 35 if market_regime == "volatile" else 40
            
            if current['RSI'] < rsi_threshold:
                return {
                    'strategy': 'Support_Bounce_Flexible',
                    'direction': 'LONG',
                    'entry': current['Close'],
                    'stop': recent_low - current['ATR'] * 0.2,
                    'target': current['Close'] + current['ATR'] * 2,
                    'confidence': calculate_flexible_confidence(current, df, current_idx, 'reversal'),
                    'regime': market_regime
                }
    
    return None

def calculate_flexible_confidence(current, df, current_idx, strategy_type):
    """Calculate confidence score with multiple factors"""
    confidence = 50  # Base confidence
    
    # Volume factor
    vol_avg = df['Volume'].rolling(20).mean().iloc[current_idx]
    vol_ratio = current['Volume'] / vol_avg
    if vol_ratio > 2:
        confidence += 15
    elif vol_ratio > 1.5:
        confidence += 10
    elif vol_ratio > 1.2:
        confidence += 5
    
    # Trend alignment
    if current['EMA_9'] > current['EMA_21']:
        confidence += 5
    
    # Time of day factor
    hour = current.name.hour + current.name.minute / 60
    if 10 <= hour <= 11:  # Best time for breakouts
        if strategy_type == "breakout":
            confidence += 10
    elif 14 <= hour <= 15:  # Good for momentum
        if strategy_type == "momentum":
            confidence += 10
    
    # Market regime bonus
    if hasattr(current, 'regime'):
        if current.regime == "trending" and strategy_type in ["momentum", "breakout"]:
            confidence += 10
        elif current.regime == "ranging" and strategy_type == "mean_reversion":
            confidence += 10
    
    # Cap confidence
    return min(confidence, 95)

def fetch_and_analyze_flexible(ticker, use_last_trading_day=False):
    """Fetch data and run flexible signal detection"""
    try:
        # Download recent data
        stock = yf.Ticker(ticker)
        df = stock.history(period="5d", interval="1m")
        
        # For Canadian stocks, try 2m if 1m is empty
        if df.empty and ticker.endswith('.TO'):
            df = stock.history(period="5d", interval="2m")
        
        if df.empty:
            return None
        
        # Calculate all indicators
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        df['VWAP'] = calculate_vwap(df)
        df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'])
        df['EMA_9'] = df['Close'].ewm(span=9).mean()
        df['EMA_21'] = df['Close'].ewm(span=21).mean()
        df['ADX'] = calculate_adx(df['High'], df['Low'], df['Close'])
        
        # Get target day's data
        if use_last_trading_day:
            available_dates = df.index.date
            unique_dates = sorted(set(available_dates), reverse=True)
            if len(unique_dates) > 0:
                target_date = unique_dates[0]
                df_today = df[df.index.date == target_date]
                print(f"  Using last trading day data: {target_date}")
            else:
                return None
        else:
            today = datetime.now().date()
            df_today = df[df.index.date == today]
        
        if len(df_today) < 30:
            if not use_last_trading_day:
                print(f"  Insufficient data for today, only {len(df_today)} bars")
            return None
        
        # Detect market regime
        market_regime = get_market_regime(df_today)
        print(f"  Market regime: {market_regime}")
        
        # Run flexible strategies on latest bar
        current_idx = len(df_today) - 1
        current_hour = df_today.index[current_idx].hour + df_today.index[current_idx].minute / 60
        
        # Try each strategy with flexible parameters
        strategies = [
            ('breakout', lambda: detect_orb_flexible(df_today, current_idx, market_regime)),
            ('mean_reversion', lambda: detect_vwap_bounce_flexible(df_today, current_idx, market_regime)),
            ('momentum', lambda: detect_ema_pullback_flexible(df_today, current_idx, market_regime)),
            ('reversal', lambda: detect_support_bounce_flexible(df_today, current_idx, market_regime))
        ]
        
        for strategy_type, detect_func in strategies:
            # Check if we should skip this strategy type
            if should_skip_strategy(strategy_type, current_hour):
                continue
            
            signal = detect_func()
            if signal and signal['confidence'] >= 78:  # Minimum confidence threshold
                signal['ticker'] = ticker
                signal['time'] = df_today.index[current_idx].strftime('%Y-%m-%d %H:%M:%S')
                return signal
        
        return None
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        return None

# Email functions (reuse from original)
def send_email_alert(signal):
    """Send email alert for detected signal"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    sender = os.environ.get('GMAIL_USER')
    password = os.environ.get('GMAIL_APP_PWD')
    recipient = os.environ.get('ALERT_TO', sender)
    
    if not sender or not password:
        print("Email credentials not configured")
        return
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = f"🎯 {signal['ticker']} - {signal['strategy']} Signal"
    
    body = f"""
Trading Signal Detected!

Strategy: {signal['strategy']}
Ticker: {signal['ticker']}
Direction: {signal['direction']}
Entry: ${signal['entry']:.2f}
Stop Loss: ${signal['stop']:.2f}
Target: ${signal['target']:.2f}
Risk/Reward: {abs(signal['target'] - signal['entry']) / abs(signal['entry'] - signal['stop']):.2f}
Confidence: {signal['confidence']}%
Market Regime: {signal.get('regime', 'Unknown')}
Time: {signal['time']}

This is a flexible parameter signal that adapts to market conditions.
"""
    
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent for {signal['ticker']} - {signal['strategy']}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def load_sent_signals():
    """Load previously sent signals to avoid duplicates"""
    try:
        with open('sent_signals.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def save_sent_signals(sent_signals):
    """Save sent signals to file"""
    with open('sent_signals.json', 'w') as f:
        json.dump(sent_signals, f)

def generate_signal_id(signal):
    """Generate unique ID for signal"""
    return f"{signal['ticker']}_{signal['strategy']}_{signal['direction']}_{signal['time'][:10]}"

def is_duplicate_signal(signal, sent_signals, hours=4):
    """Check if signal was already sent recently"""
    signal_id = generate_signal_id(signal)
    
    if signal_id in sent_signals:
        sent_time = datetime.fromisoformat(sent_signals[signal_id])
        current_time = datetime.fromisoformat(signal['time'])
        
        if (current_time - sent_time).total_seconds() < hours * 3600:
            return True
    
    return False

def main():
    """Main function for GitHub Actions"""
    print(f"Starting Flexible Signal Detection - {datetime.now()}")
    
    # Get tickers from environment or use defaults
    tickers_env = os.environ.get('TICKERS', 'NVDA,AAPL,MSFT,TSLA,META,GOOGL,AMZN,AMD,NFLX,SPY')
    tickers = [t.strip() for t in tickers_env.split(',')]
    
    # Check if in test mode
    test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
    use_last_trading_day = test_mode
    
    print(f"Scanning {len(tickers)} tickers with flexible parameters")
    print(f"Test mode: {test_mode}")
    
    # Load sent signals
    sent_signals = load_sent_signals()
    
    # Results
    all_signals = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        signal = fetch_and_analyze_flexible(ticker, use_last_trading_day)
        
        if signal:
            print(f"  ✓ Signal detected: {signal['strategy']} @ ${signal['entry']:.2f}")
            all_signals.append(signal)
            
            # Check for duplicate
            if not is_duplicate_signal(signal, sent_signals):
                if not test_mode:
                    send_email_alert(signal)
                    sent_signals[generate_signal_id(signal)] = signal['time']
            else:
                print(f"  → Duplicate signal, skipping email")
        else:
            print(f"  → No signals detected")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'flexible_parameters',
        'tickers_scanned': len(tickers),
        'signals': all_signals,
        'test_mode': test_mode
    }
    
    with open('signals.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save sent signals
    if not test_mode:
        save_sent_signals(sent_signals)
    
    print(f"\nScan complete: {len(all_signals)} signals found")
    
    return 0 if all_signals else 1

if __name__ == "__main__":
    sys.exit(main())