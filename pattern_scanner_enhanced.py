import yfinance as yf
import pandas as pd
import numpy as np
from tradingpatterns import TradingPatternScanner
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def clean_yfinance_data(data):
    """
    Cleans data from yfinance, handling potential multi-index columns
    and ensuring data is numeric.
    """
    cleaned_data = data.copy()
    
    # Handle multi-index columns if present
    if isinstance(cleaned_data.columns, pd.MultiIndex):
        cleaned_data.columns = cleaned_data.columns.get_level_values(0)
    
    # Ensure required columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    # Remove any NaN values
    cleaned_data.dropna(inplace=True)
    
    # Ensure index is datetime
    if not isinstance(cleaned_data.index, pd.DatetimeIndex):
        cleaned_data.index = pd.to_datetime(cleaned_data.index)
    
    return cleaned_data

def scan_with_tradingpatterns(data, method='wavelet'):
    """
    Scan for patterns using TradingPatternScanner library
    
    Methods available:
    - 'window': Rolling window extrema (78.5% accuracy)
    - 'savgol': Savitzky-Golay filter (78.5% accuracy)
    - 'kalman': Kalman filter (73.5% accuracy)
    - 'wavelet': Wavelet denoising (84.5% accuracy) - RECOMMENDED
    """
    scanner = TradingPatternScanner(method=method)
    
    # Prepare data in the format expected by the library
    prices = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # Scan for patterns
    patterns = scanner.scan_all_patterns(prices)
    
    return patterns

def find_head_and_shoulders(data):
    """Find head and shoulders pattern using TradingPatternScanner"""
    try:
        scanner = TradingPatternScanner(method='wavelet')
        patterns = scanner.find_head_and_shoulders(data)
        
        if patterns and len(patterns) > 0:
            # Return the most recent pattern
            latest_pattern = patterns[-1]
            # Convert to format expected by calculate_trading_signals
            return (latest_pattern['left_shoulder_idx'], 
                   latest_pattern['head_idx'], 
                   latest_pattern['right_shoulder_idx'],
                   latest_pattern['left_trough_idx'],
                   latest_pattern['right_trough_idx'])
    except Exception as e:
        print(f"Error in head_and_shoulders detection: {e}")
    
    return None

def find_inverse_head_and_shoulders(data):
    """Find inverse head and shoulders pattern using TradingPatternScanner"""
    try:
        scanner = TradingPatternScanner(method='wavelet')
        patterns = scanner.find_inverse_head_and_shoulders(data)
        
        if patterns and len(patterns) > 0:
            # Return the most recent pattern
            latest_pattern = patterns[-1]
            return (latest_pattern['left_shoulder_idx'], 
                   latest_pattern['head_idx'], 
                   latest_pattern['right_shoulder_idx'],
                   latest_pattern['left_peak_idx'],
                   latest_pattern['right_peak_idx'])
    except Exception as e:
        print(f"Error in inverse_head_and_shoulders detection: {e}")
    
    return None

def find_double_top(data):
    """Find double top pattern using TradingPatternScanner"""
    try:
        scanner = TradingPatternScanner(method='wavelet')
        patterns = scanner.find_double_tops(data)
        
        if patterns and len(patterns) > 0:
            # Return the most recent pattern
            latest_pattern = patterns[-1]
            return (latest_pattern['first_peak_idx'], 
                   latest_pattern['second_peak_idx'], 
                   latest_pattern['valley_idx'])
    except Exception as e:
        print(f"Error in double_top detection: {e}")
    
    return None

def find_double_bottom(data):
    """Find double bottom pattern using TradingPatternScanner"""
    try:
        scanner = TradingPatternScanner(method='wavelet')
        patterns = scanner.find_double_bottoms(data)
        
        if patterns and len(patterns) > 0:
            # Return the most recent pattern
            latest_pattern = patterns[-1]
            return (latest_pattern['first_trough_idx'], 
                   latest_pattern['second_trough_idx'], 
                   latest_pattern['peak_idx'])
    except Exception as e:
        print(f"Error in double_bottom detection: {e}")
    
    return None

def calculate_trading_signals(data, pattern_name, points):
    """Calculate actionable trading signals based on the pattern"""
    current_price = float(data['Close'].iloc[-1])
    signals = {}
    
    if pattern_name == "Head and Shoulders":
        l, p, r, lt, rt = points
        # Ensure indices are integers
        l, p, r, lt, rt = int(l), int(p), int(r), int(lt), int(rt)
        
        # Neckline is the average of the two troughs
        neckline = (data['Low'].iloc[lt] + data['Low'].iloc[rt]) / 2
        head_height = data['High'].iloc[p] - neckline
        
        signals['action'] = 'SELL'
        signals['entry'] = round(float(neckline * 0.99), 2)  # Enter on break below neckline
        signals['stop_loss'] = round(float(data['High'].iloc[r] * 1.01), 2)  # Above right shoulder
        signals['target'] = round(float(neckline - head_height), 2)  # Project head height below neckline
        
    elif pattern_name == "Inverse Head and Shoulders":
        l, h, r, lp, rp = points
        l, h, r, lp, rp = int(l), int(h), int(r), int(lp), int(rp)
        
        # Neckline is the average of the two peaks
        neckline = (data['High'].iloc[lp] + data['High'].iloc[rp]) / 2
        head_height = neckline - data['Low'].iloc[h]
        
        signals['action'] = 'BUY'
        signals['entry'] = round(float(neckline * 1.01), 2)  # Enter on break above neckline
        signals['stop_loss'] = round(float(data['Low'].iloc[r] * 0.99), 2)  # Below right shoulder
        signals['target'] = round(float(neckline + head_height), 2)  # Project head height above neckline
        
    elif pattern_name == "Double Top":
        p1, p2, valley = points
        p1, p2, valley = int(p1), int(p2), int(valley)
        
        resistance = (data['High'].iloc[p1] + data['High'].iloc[p2]) / 2
        support = data['Low'].iloc[valley]
        pattern_height = resistance - support
        
        signals['action'] = 'SELL'
        signals['entry'] = round(float(support * 0.99), 2)  # Enter on break below valley
        signals['stop_loss'] = round(float(resistance * 1.01), 2)  # Above the double top
        signals['target'] = round(float(support - pattern_height * 0.8), 2)  # Project 80% of pattern height
        
    elif pattern_name == "Double Bottom":
        t1, t2, peak = points
        t1, t2, peak = int(t1), int(t2), int(peak)
        
        support = (data['Low'].iloc[t1] + data['Low'].iloc[t2]) / 2
        resistance = data['High'].iloc[peak]
        pattern_height = resistance - support
        
        signals['action'] = 'BUY'
        signals['entry'] = round(float(resistance * 1.01), 2)  # Enter on break above peak
        signals['stop_loss'] = round(float(support * 0.99), 2)  # Below the double bottom
        signals['target'] = round(float(resistance + pattern_height * 0.8), 2)  # Project 80% of pattern height
    
    # Calculate risk/reward ratio
    if 'entry' in signals and 'stop_loss' in signals and 'target' in signals:
        risk = abs(signals['entry'] - signals['stop_loss'])
        reward = abs(signals['target'] - signals['entry'])
        signals['risk_reward'] = round(reward / risk if risk > 0 else 0, 2)
    
    signals['current_price'] = round(current_price, 2)
    signals['distance_to_entry'] = round(abs(current_price - signals['entry']) / current_price * 100, 2)
    
    # Add pattern confidence based on TradingPatternScanner's accuracy
    signals['confidence'] = 84.5  # Wavelet method accuracy
    
    return signals

def calculate_pattern_confidence(data, pattern_name, points):
    """
    Calculate confidence score for the pattern
    Using TradingPatternScanner's reported accuracy for wavelet method
    """
    # Base confidence from wavelet method accuracy
    confidence = 84.5
    
    # Additional adjustments based on pattern recency
    if pattern_name in ["Double Top", "Double Bottom"]:
        if pattern_name == "Double Top":
            p1, p2, valley = points
            pattern_end = max(p1, p2)
        else:  # Double Bottom
            t1, t2, peak = points
            pattern_end = max(t1, t2)
        
        # Reduce confidence if pattern is too recent (less reliable)
        pattern_age = len(data) - pattern_end
        if pattern_age < 5:  # Very recent pattern
            confidence -= 10
        elif pattern_age > 20:  # Well-formed pattern
            confidence += 5
    
    # Ensure confidence is between 0 and 100
    confidence = max(0, min(100, confidence))
    
    return round(confidence, 1)

def plot_pattern(data, pattern_name, points, ticker):
    """
    Visualization function (optional)
    Note: Commented out for GitHub Actions compatibility
    """
    pass

# Fallback functions for compatibility
def scan_patterns(ticker, data=None, interval='5m', period='1d'):
    """
    Main function to scan for all patterns
    """
    if data is None:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if data.empty:
        return None
    
    cleaned_data = clean_yfinance_data(data)
    
    patterns = {
        "Head and Shoulders": find_head_and_shoulders,
        "Inverse Head and Shoulders": find_inverse_head_and_shoulders,
        "Double Top": find_double_top,
        "Double Bottom": find_double_bottom
    }
    
    results = {}
    for pattern_name, find_func in patterns.items():
        result = find_func(cleaned_data)
        if result is not None:
            signals = calculate_trading_signals(cleaned_data, pattern_name, result)
            results[pattern_name] = {
                'pattern_points': result,
                'signals': signals,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return results

if __name__ == "__main__":
    # Test the scanner
    ticker = "NVDA"
    results = scan_patterns(ticker)
    
    if results:
        print(f"\nPatterns found for {ticker}:")
        for pattern_name, pattern_data in results.items():
            print(f"\n{pattern_name}:")
            print(f"  Action: {pattern_data['signals']['action']}")
            print(f"  Entry: ${pattern_data['signals']['entry']}")
            print(f"  Stop Loss: ${pattern_data['signals']['stop_loss']}")
            print(f"  Target: ${pattern_data['signals']['target']}")
            print(f"  Risk/Reward: {pattern_data['signals']['risk_reward']}")
            print(f"  Confidence: {pattern_data['signals']['confidence']}%")
    else:
        print(f"No patterns found for {ticker}")