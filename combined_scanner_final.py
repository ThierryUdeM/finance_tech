#!/usr/bin/env python3
"""
Combined Pattern Scanner using both TradingPattern and TA-Lib
Run with: python3 combined_scanner_final.py
"""

import sys
# Add the working TA-Lib path
sys.path.insert(0, '/home/thierrygc/script/pattern_detection/venv/lib/python3.12/site-packages')

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import tradingpatterns (may need virtual env)
try:
    from tradingpatterns.tradingpatterns import (
        detect_head_shoulder,
        detect_double_top_bottom
    )
    HAS_TRADING_PATTERNS = True
    print("✓ Using tradingpattern library (84.5% accuracy)")
except ImportError:
    HAS_TRADING_PATTERNS = False
    print("⚠ tradingpattern library not available - chart patterns disabled")

print(f"✓ Using TA-Lib version {talib.__version__} for candlestick patterns")

def clean_data(data):
    """Clean and prepare data for analysis"""
    cleaned = data.copy()
    
    # Handle multi-index columns
    if isinstance(cleaned.columns, pd.MultiIndex):
        cleaned.columns = cleaned.columns.get_level_values(0)
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    
    cleaned.dropna(inplace=True)
    
    # Ensure datetime index
    if not isinstance(cleaned.index, pd.DatetimeIndex):
        cleaned.index = pd.to_datetime(cleaned.index)
    
    return cleaned

def detect_candlestick_patterns(data):
    """Detect candlestick patterns using TA-Lib"""
    patterns = {}
    
    # Extract OHLC arrays
    open_prices = data['Open'].values
    high_prices = data['High'].values
    low_prices = data['Low'].values
    close_prices = data['Close'].values
    
    # Key candlestick patterns for intraday trading
    key_patterns = {
        'CDLDOJI': ('Doji', 'Indecision'),
        'CDLHAMMER': ('Hammer', 'Bullish reversal'),
        'CDLINVERTEDHAMMER': ('Inverted Hammer', 'Bullish reversal'),
        'CDLSHOOTINGSTAR': ('Shooting Star', 'Bearish reversal'),
        'CDLENGULFING': ('Engulfing Pattern', 'Strong reversal'),
        'CDLHARAMI': ('Harami', 'Potential reversal'),
        'CDLMORNINGSTAR': ('Morning Star', 'Bullish reversal'),
        'CDLEVENINGSTAR': ('Evening Star', 'Bearish reversal'),
        'CDLSPINNINGTOP': ('Spinning Top', 'Indecision'),
        'CDLMARUBOZU': ('Marubozu', 'Strong trend'),
        'CDLDRAGONFLYDOJI': ('Dragonfly Doji', 'Bullish reversal'),
        'CDLGRAVESTONEDOJI': ('Gravestone Doji', 'Bearish reversal'),
        'CDLPIERCING': ('Piercing Pattern', 'Bullish reversal'),
        'CDLDARKCLOUDCOVER': ('Dark Cloud Cover', 'Bearish reversal'),
        'CDL3WHITESOLDIERS': ('Three White Soldiers', 'Strong bullish'),
        'CDL3BLACKCROWS': ('Three Black Crows', 'Strong bearish')
    }
    
    # Detect patterns
    for func_name, (pattern_name, description) in key_patterns.items():
        try:
            pattern_func = getattr(talib, func_name)
            result = pattern_func(open_prices, high_prices, low_prices, close_prices)
            
            # Find where pattern occurs (non-zero values)
            pattern_indices = np.where(result != 0)[0]
            
            if len(pattern_indices) > 0:
                patterns[pattern_name] = []
                for idx in pattern_indices:
                    patterns[pattern_name].append({
                        'index': data.index[idx],
                        'signal': int(result[idx]),  # 100 for bullish, -100 for bearish
                        'price': float(data['Close'].iloc[idx]),
                        'description': description
                    })
        except Exception as e:
            continue
    
    return patterns

def detect_chart_patterns(data):
    """Detect chart patterns using tradingpattern library"""
    patterns = {}
    
    if not HAS_TRADING_PATTERNS:
        return patterns
    
    try:
        # Head and Shoulders
        df_hs = detect_head_shoulder(data.copy(), window=5)
        hs_patterns = df_hs[df_hs['head_shoulder_pattern'].notna()]
        
        if len(hs_patterns) > 0:
            patterns['Head and Shoulders'] = []
            patterns['Inverse Head and Shoulders'] = []
            
            for idx, row in hs_patterns.iterrows():
                pattern_type = row['head_shoulder_pattern']
                pattern_data = {
                    'index': idx,
                    'price': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'description': 'Bearish reversal' if 'Inverse' not in pattern_type else 'Bullish reversal'
                }
                
                if 'Inverse' in pattern_type:
                    patterns['Inverse Head and Shoulders'].append(pattern_data)
                else:
                    patterns['Head and Shoulders'].append(pattern_data)
        
        # Double Tops/Bottoms
        df_dt = detect_double_top_bottom(data.copy(), window=5, threshold=0.02)
        dt_patterns = df_dt[df_dt['double_pattern'].notna()]
        
        if len(dt_patterns) > 0:
            patterns['Double Top'] = []
            patterns['Double Bottom'] = []
            
            for idx, row in dt_patterns.iterrows():
                pattern_type = row['double_pattern']
                pattern_data = {
                    'index': idx,
                    'price': float(row['Close']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'description': 'Bearish reversal' if 'Top' in pattern_type else 'Bullish reversal'
                }
                
                if 'Top' in pattern_type:
                    patterns['Double Top'].append(pattern_data)
                else:
                    patterns['Double Bottom'].append(pattern_data)
                    
    except Exception as e:
        print(f"Error detecting chart patterns: {e}")
    
    # Clean up empty pattern lists
    patterns = {k: v for k, v in patterns.items() if v}
    
    return patterns

def analyze_patterns(ticker, period='1d', interval='5m'):
    """Main function to analyze all patterns"""
    print(f"\n{'='*60}")
    print(f"Combined Pattern Analysis for {ticker}")
    print(f"{'='*60}")
    
    # Download data
    print(f"\nDownloading {ticker} data...")
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if data.empty:
        print("Error: No data retrieved")
        return None
    
    # Clean data
    data = clean_data(data)
    
    print(f"Data period: {data.index[0]} to {data.index[-1]}")
    print(f"Total candles: {len(data)}")
    print(f"Current price: ${float(data['Close'].iloc[-1]):.2f}")
    
    # Detect candlestick patterns
    print(f"\n{'='*40}")
    print("CANDLESTICK PATTERNS (TA-Lib)")
    print('='*40)
    
    candlestick_patterns = detect_candlestick_patterns(data)
    
    if candlestick_patterns:
        # Group by bullish/bearish
        bullish_candles = []
        bearish_candles = []
        neutral_candles = []
        
        for pattern_name, occurrences in candlestick_patterns.items():
            latest = occurrences[-1] if occurrences else None
            if latest:
                if latest['signal'] > 0:
                    bullish_candles.append((pattern_name, latest))
                elif latest['signal'] < 0:
                    bearish_candles.append((pattern_name, latest))
                else:
                    neutral_candles.append((pattern_name, latest))
        
        if bullish_candles:
            print("\n🟢 Bullish Signals:")
            for pattern_name, occ in bullish_candles:
                print(f"  • {pattern_name}: {occ['index'].strftime('%H:%M')} @ ${occ['price']:.2f}")
                print(f"    ({occ['description']})")
        
        if bearish_candles:
            print("\n🔴 Bearish Signals:")
            for pattern_name, occ in bearish_candles:
                print(f"  • {pattern_name}: {occ['index'].strftime('%H:%M')} @ ${occ['price']:.2f}")
                print(f"    ({occ['description']})")
        
        if neutral_candles:
            print("\n⚪ Neutral/Indecision:")
            for pattern_name, occ in neutral_candles:
                print(f"  • {pattern_name}: {occ['index'].strftime('%H:%M')} @ ${occ['price']:.2f}")
    else:
        print("No candlestick patterns detected")
    
    # Detect chart patterns
    print(f"\n{'='*40}")
    print("CHART PATTERNS (TradingPattern)")
    print('='*40)
    
    chart_patterns = detect_chart_patterns(data)
    
    if chart_patterns:
        for pattern_name, occurrences in chart_patterns.items():
            print(f"\n{pattern_name}:")
            latest = occurrences[-1] if occurrences else None
            if latest:
                print(f"  • Latest: {latest['index'].strftime('%H:%M')} @ ${latest['price']:.2f}")
                print(f"    ({latest['description']})")
                print(f"    High: ${latest['high']:.2f}, Low: ${latest['low']:.2f}")
    else:
        print("No chart patterns detected")
    
    # Summary and recommendation
    print(f"\n{'='*40}")
    print("SUMMARY & MARKET BIAS")
    print('='*40)
    
    # Count signals
    bullish_count = 0
    bearish_count = 0
    
    # Count candlestick signals
    for patterns in candlestick_patterns.values():
        for p in patterns:
            if p['signal'] > 0:
                bullish_count += 1
            elif p['signal'] < 0:
                bearish_count += 1
    
    # Count chart pattern signals
    if chart_patterns:
        bullish_count += len(chart_patterns.get('Inverse Head and Shoulders', []))
        bullish_count += len(chart_patterns.get('Double Bottom', []))
        bearish_count += len(chart_patterns.get('Head and Shoulders', []))
        bearish_count += len(chart_patterns.get('Double Top', []))
    
    total_patterns = sum(len(v) for v in candlestick_patterns.values()) + \
                    sum(len(v) for v in chart_patterns.values())
    
    print(f"\nTotal patterns detected: {total_patterns}")
    print(f"  • Candlestick: {sum(len(v) for v in candlestick_patterns.values())}")
    print(f"  • Chart: {sum(len(v) for v in chart_patterns.values())}")
    
    print(f"\nSignal distribution:")
    print(f"  • Bullish: {bullish_count}")
    print(f"  • Bearish: {bearish_count}")
    
    # Market bias
    if total_patterns > 0:
        bullish_pct = (bullish_count / (bullish_count + bearish_count) * 100) if (bullish_count + bearish_count) > 0 else 0
        
        print(f"\n📊 Market Bias: ", end="")
        if bullish_pct > 60:
            print(f"BULLISH ({bullish_pct:.0f}%)")
            print("   Recommendation: Look for long opportunities")
        elif bullish_pct < 40:
            print(f"BEARISH ({100-bullish_pct:.0f}%)")
            print("   Recommendation: Look for short opportunities")
        else:
            print(f"NEUTRAL ({bullish_pct:.0f}% bullish)")
            print("   Recommendation: Wait for clearer signals")
    
    return {
        'candlestick_patterns': candlestick_patterns,
        'chart_patterns': chart_patterns,
        'bullish_count': bullish_count,
        'bearish_count': bearish_count
    }

if __name__ == "__main__":
    # Test with NVDA
    results = analyze_patterns("NVDA", period='1d', interval='5m')