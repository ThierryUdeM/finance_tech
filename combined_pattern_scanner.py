#!/usr/bin/env python3
"""
Combined Pattern Scanner using both TradingPattern and TA-Lib
Provides comprehensive pattern detection:
- Chart patterns (Head & Shoulders, Double Tops, etc.) via tradingpattern
- Candlestick patterns (Doji, Hammer, etc.) via TA-Lib
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from tradingpatterns.tradingpatterns import (
        detect_head_shoulder,
        detect_double_top_bottom,
        detect_triangle_pattern,
        detect_wedge
    )
    HAS_TRADING_PATTERNS = True
except ImportError:
    HAS_TRADING_PATTERNS = False
    print("Warning: tradingpattern library not available")

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
    
    # Define all available candlestick pattern functions
    candlestick_patterns = {
        'CDL2CROWS': 'Two Crows',
        'CDL3BLACKCROWS': 'Three Black Crows',
        'CDL3INSIDE': 'Three Inside Up/Down',
        'CDL3LINESTRIKE': 'Three-Line Strike',
        'CDL3OUTSIDE': 'Three Outside Up/Down',
        'CDL3STARSINSOUTH': 'Three Stars In The South',
        'CDL3WHITESOLDIERS': 'Three Advancing White Soldiers',
        'CDLABANDONEDBABY': 'Abandoned Baby',
        'CDLADVANCEBLOCK': 'Advance Block',
        'CDLBELTHOLD': 'Belt-hold',
        'CDLBREAKAWAY': 'Breakaway',
        'CDLCLOSINGMARUBOZU': 'Closing Marubozu',
        'CDLCONCEALBABYSWALL': 'Concealing Baby Swallow',
        'CDLCOUNTERATTACK': 'Counterattack',
        'CDLDARKCLOUDCOVER': 'Dark Cloud Cover',
        'CDLDOJI': 'Doji',
        'CDLDOJISTAR': 'Doji Star',
        'CDLDRAGONFLYDOJI': 'Dragonfly Doji',
        'CDLENGULFING': 'Engulfing Pattern',
        'CDLEVENINGDOJISTAR': 'Evening Doji Star',
        'CDLEVENINGSTAR': 'Evening Star',
        'CDLGAPSIDESIDEWHITE': 'Up/Down-gap side-by-side white lines',
        'CDLGRAVESTONEDOJI': 'Gravestone Doji',
        'CDLHAMMER': 'Hammer',
        'CDLHANGINGMAN': 'Hanging Man',
        'CDLHARAMI': 'Harami Pattern',
        'CDLHARAMICROSS': 'Harami Cross Pattern',
        'CDLHIGHWAVE': 'High-Wave Candle',
        'CDLHIKKAKE': 'Hikkake Pattern',
        'CDLHIKKAKEMOD': 'Modified Hikkake Pattern',
        'CDLHOMINGPIGEON': 'Homing Pigeon',
        'CDLIDENTICAL3CROWS': 'Identical Three Crows',
        'CDLINNECK': 'In-Neck Pattern',
        'CDLINVERTEDHAMMER': 'Inverted Hammer',
        'CDLKICKING': 'Kicking',
        'CDLKICKINGBYLENGTH': 'Kicking - bull/bear determined by the longer marubozu',
        'CDLLADDERBOTTOM': 'Ladder Bottom',
        'CDLLONGLEGGEDDOJI': 'Long Legged Doji',
        'CDLLONGLINE': 'Long Line Candle',
        'CDLMARUBOZU': 'Marubozu',
        'CDLMATCHINGLOW': 'Matching Low',
        'CDLMATHOLD': 'Mat Hold',
        'CDLMORNINGDOJISTAR': 'Morning Doji Star',
        'CDLMORNINGSTAR': 'Morning Star',
        'CDLONNECK': 'On-Neck Pattern',
        'CDLPIERCING': 'Piercing Pattern',
        'CDLRICKSHAWMAN': 'Rickshaw Man',
        'CDLRISEFALL3METHODS': 'Rising/Falling Three Methods',
        'CDLSEPARATINGLINES': 'Separating Lines',
        'CDLSHOOTINGSTAR': 'Shooting Star',
        'CDLSHORTLINE': 'Short Line Candle',
        'CDLSPINNINGTOP': 'Spinning Top',
        'CDLSTALLEDPATTERN': 'Stalled Pattern',
        'CDLSTICKSANDWICH': 'Stick Sandwich',
        'CDLTAKURI': 'Takuri (Dragonfly Doji with very long lower shadow)',
        'CDLTASUKIGAP': 'Tasuki Gap',
        'CDLTHRUSTING': 'Thrusting Pattern',
        'CDLTRISTAR': 'Tristar Pattern',
        'CDLUNIQUE3RIVER': 'Unique 3 River',
        'CDLUPSIDEGAP2CROWS': 'Upside Gap Two Crows',
        'CDLXSIDEGAP3METHODS': 'Upside/Downside Gap Three Methods'
    }
    
    # Detect patterns
    for func_name, pattern_name in candlestick_patterns.items():
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
                        'price': float(data['Close'].iloc[idx])
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
                    'low': float(row['Low'])
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
                    'low': float(row['Low'])
                }
                
                if 'Top' in pattern_type:
                    patterns['Double Top'].append(pattern_data)
                else:
                    patterns['Double Bottom'].append(pattern_data)
        
        # Triangle Patterns
        df_triangle = detect_triangle_pattern(data.copy())
        triangle_patterns = df_triangle[df_triangle['triangle_pattern'].notna()]
        
        if len(triangle_patterns) > 0:
            patterns['Triangle'] = []
            for idx, row in triangle_patterns.iterrows():
                patterns['Triangle'].append({
                    'index': idx,
                    'type': row['triangle_pattern'],
                    'price': float(row['Close'])
                })
        
        # Wedge Patterns
        df_wedge = detect_wedge(data.copy())
        wedge_patterns = df_wedge[df_wedge['wedge_pattern'].notna()]
        
        if len(wedge_patterns) > 0:
            patterns['Wedge'] = []
            for idx, row in wedge_patterns.iterrows():
                patterns['Wedge'].append({
                    'index': idx,
                    'type': row['wedge_pattern'],
                    'price': float(row['Close'])
                })
                
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
    
    results = {
        'ticker': ticker,
        'timestamp': datetime.now(),
        'current_price': float(data['Close'].iloc[-1]),
        'data_period': f"{data.index[0]} to {data.index[-1]}",
        'candlestick_patterns': {},
        'chart_patterns': {}
    }
    
    # Detect candlestick patterns
    print(f"\n{'='*40}")
    print("CANDLESTICK PATTERNS (TA-Lib)")
    print('='*40)
    
    candlestick_patterns = detect_candlestick_patterns(data)
    results['candlestick_patterns'] = candlestick_patterns
    
    if candlestick_patterns:
        for pattern_name, occurrences in candlestick_patterns.items():
            print(f"\n{pattern_name}:")
            for occ in occurrences[-3:]:  # Show last 3 occurrences
                signal_type = "Bullish" if occ['signal'] > 0 else "Bearish"
                print(f"  - {occ['index']}: {signal_type} (${occ['price']:.2f})")
    else:
        print("No candlestick patterns detected")
    
    # Detect chart patterns
    print(f"\n{'='*40}")
    print("CHART PATTERNS (TradingPattern)")
    print('='*40)
    
    chart_patterns = detect_chart_patterns(data)
    results['chart_patterns'] = chart_patterns
    
    if chart_patterns:
        for pattern_name, occurrences in chart_patterns.items():
            print(f"\n{pattern_name}:")
            for occ in occurrences[-3:]:  # Show last 3 occurrences
                print(f"  - {occ['index']}: ${occ['price']:.2f}")
    else:
        print("No chart patterns detected")
    
    # Summary
    print(f"\n{'='*40}")
    print("PATTERN SUMMARY")
    print('='*40)
    
    total_candlestick = sum(len(v) for v in candlestick_patterns.values())
    total_chart = sum(len(v) for v in chart_patterns.values())
    
    print(f"Total candlestick patterns: {total_candlestick}")
    print(f"Total chart patterns: {total_chart}")
    print(f"Combined total: {total_candlestick + total_chart}")
    
    # Bullish vs Bearish signals
    bullish_count = sum(1 for p in candlestick_patterns.values() 
                       for occ in p if occ['signal'] > 0)
    bearish_count = sum(1 for p in candlestick_patterns.values() 
                       for occ in p if occ['signal'] < 0)
    
    if chart_patterns:
        # Count chart patterns as bullish/bearish
        bullish_chart = len(chart_patterns.get('Inverse Head and Shoulders', [])) + \
                       len(chart_patterns.get('Double Bottom', []))
        bearish_chart = len(chart_patterns.get('Head and Shoulders', [])) + \
                       len(chart_patterns.get('Double Top', []))
        bullish_count += bullish_chart
        bearish_count += bearish_chart
    
    print(f"\nSignal Distribution:")
    print(f"  Bullish signals: {bullish_count}")
    print(f"  Bearish signals: {bearish_count}")
    
    if bullish_count > bearish_count:
        print(f"  Overall bias: BULLISH ({bullish_count}/{bullish_count+bearish_count})")
    elif bearish_count > bullish_count:
        print(f"  Overall bias: BEARISH ({bearish_count}/{bullish_count+bearish_count})")
    else:
        print(f"  Overall bias: NEUTRAL")
    
    return results

if __name__ == "__main__":
    # Test with NVDA
    results = analyze_patterns("NVDA", period='1d', interval='5m')
    
    # You can also try with longer periods
    # results = analyze_patterns("NVDA", period='5d', interval='15m')