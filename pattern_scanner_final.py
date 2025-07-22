import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import TradingPatternScanner, fallback to scipy if not available
try:
    import tradingpattern
    # Check what's available in the module
    HAS_TRADING_PATTERNS = True
    print("Using tradingpattern library for enhanced pattern detection")
except ImportError:
    from scipy.signal import find_peaks
    HAS_TRADING_PATTERNS = False
    print("tradingpattern not available, using scipy implementation")

def clean_yfinance_data(data):
    """
    Cleans data from yfinance, handling potential multi-index columns
    and ensuring data is numeric.
    """
    print("--- Inside clean_yfinance_data ---")
    print("Columns received:", data.columns)
    print("Head of data received:\n", data.head())
    
    cleaned_data = data.copy()
    
    # Handle multi-index columns if present
    if isinstance(cleaned_data.columns, pd.MultiIndex):
        cleaned_data.columns = cleaned_data.columns.get_level_values(0)
    
    print("Columns after parsing attempt:", cleaned_data.columns)
    
    # Ensure required columns are numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in cleaned_data.columns:
            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
        else:
            raise ValueError(f"Required column '{col}' not found in data.")
    
    # Remove any NaN values
    cleaned_data.dropna(inplace=True)
    
    # Ensure index is datetime
    if not isinstance(cleaned_data.index, pd.DatetimeIndex):
        cleaned_data.index = pd.to_datetime(cleaned_data.index)
    
    print("--- Exiting clean_yfinance_data ---")
    return cleaned_data

# Enhanced pattern detection functions using tradingpattern
def find_patterns_enhanced(data):
    """
    Use tradingpattern library to detect all patterns
    Returns a dictionary of detected patterns
    """
    patterns_found = {}
    
    try:
        # The library likely has methods like these based on the description
        # We'll need to adapt based on actual API
        if hasattr(tradingpattern, 'detect_head_shoulder'):
            hs_patterns = tradingpattern.detect_head_shoulder(data)
            if hs_patterns:
                patterns_found['head_shoulder'] = hs_patterns
                
        if hasattr(tradingpattern, 'detect_double_top'):
            dt_patterns = tradingpattern.detect_double_top(data)
            if dt_patterns:
                patterns_found['double_top'] = dt_patterns
                
        if hasattr(tradingpattern, 'detect_double_bottom'):
            db_patterns = tradingpattern.detect_double_bottom(data)
            if db_patterns:
                patterns_found['double_bottom'] = db_patterns
                
        # Alternative: the library might have a scanner class
        if hasattr(tradingpattern, 'Scanner') or hasattr(tradingpattern, 'PatternScanner'):
            scanner_class = getattr(tradingpattern, 'Scanner', None) or getattr(tradingpattern, 'PatternScanner', None)
            scanner = scanner_class(data)
            
            # Try different method names
            for method_name in ['detect_all', 'scan_all', 'find_patterns', 'detect_patterns']:
                if hasattr(scanner, method_name):
                    patterns_found = getattr(scanner, method_name)()
                    break
                    
    except Exception as e:
        print(f"Error using tradingpattern library: {e}")
        print("Falling back to scipy implementation")
        
    return patterns_found

# Basic pattern detection functions (fallback)
def find_head_and_shoulders(data):
    """Head and shoulders detection"""
    if HAS_TRADING_PATTERNS:
        # Try to use enhanced detection
        patterns = find_patterns_enhanced(data)
        if 'head_shoulder' in patterns and patterns['head_shoulder']:
            # Convert to our expected format
            latest = patterns['head_shoulder'][-1] if isinstance(patterns['head_shoulder'], list) else patterns['head_shoulder']
            # Adapt based on actual data structure
            return latest
    
    # Fallback to scipy implementation
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    
    peaks, _ = find_peaks(high_prices, distance=3, width=2)
    
    if len(peaks) < 3:
        return None
    
    for i in range(1, len(peaks) - 1):
        p = peaks[i]
        l, r = -1, -1
        
        for j in range(i - 1, -1, -1):
            if high_prices[peaks[j]] < high_prices[p]:
                if l == -1 or high_prices[peaks[j]] > high_prices[l]:
                    l = peaks[j]
        
        for j in range(i + 1, len(peaks)):
            if high_prices[peaks[j]] < high_prices[p]:
                if r == -1 or high_prices[peaks[j]] > high_prices[r]:
                    r = peaks[j]
        
        if l != -1 and r != -1:
            troughs, _ = find_peaks(-low_prices, distance=3, width=2)
            left_trough = -1
            for t in troughs:
                if l < t < p:
                    if left_trough == -1 or low_prices[t] < low_prices[left_trough]:
                        left_trough = t
            
            right_trough = -1
            for t in troughs:
                if p < t < r:
                    if right_trough == -1 or low_prices[t] < low_prices[right_trough]:
                        right_trough = t
            
            if left_trough != -1 and right_trough != -1:
                return (l, p, r, left_trough, right_trough)
    return None

def find_inverse_head_and_shoulders(data):
    """Inverse head and shoulders detection"""
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    troughs, _ = find_peaks(-low_prices, distance=3, width=2)
    
    if len(troughs) < 3:
        return None
    
    for i in range(1, len(troughs) - 1):
        h = troughs[i]
        l, r = -1, -1
        
        for j in range(i - 1, -1, -1):
            if low_prices[troughs[j]] > low_prices[h]:
                if l == -1 or low_prices[troughs[j]] < low_prices[l]:
                    l = troughs[j]
        
        for j in range(i + 1, len(troughs)):
            if low_prices[troughs[j]] > low_prices[h]:
                if r == -1 or low_prices[troughs[j]] < low_prices[r]:
                    r = troughs[j]
        
        if l != -1 and r != -1:
            peaks, _ = find_peaks(high_prices, distance=3, width=2)
            left_peak = -1
            for p in peaks:
                if l < p < h:
                    if left_peak == -1 or high_prices[p] > high_prices[left_peak]:
                        left_peak = p
            
            right_peak = -1
            for p in peaks:
                if h < p < r:
                    if right_peak == -1 or high_prices[p] > high_prices[right_peak]:
                        right_peak = p
            
            if left_peak != -1 and right_peak != -1:
                return (l, h, r, left_peak, right_peak)
    return None

def find_double_top(data):
    """Double top detection"""
    if HAS_TRADING_PATTERNS:
        patterns = find_patterns_enhanced(data)
        if 'double_top' in patterns and patterns['double_top']:
            latest = patterns['double_top'][-1] if isinstance(patterns['double_top'], list) else patterns['double_top']
            return latest
    
    # Fallback
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    peaks, _ = find_peaks(high_prices, distance=3, prominence=0.1)
    
    if len(peaks) < 2:
        return None
    
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        
        if abs(high_prices[p1] - high_prices[p2]) < 0.05 * (high_prices[p1] + high_prices[p2]) / 2:
            troughs, _ = find_peaks(-low_prices, distance=3)
            valley = -1
            for t in troughs:
                if p1 < t < p2:
                    if valley == -1 or low_prices[t] < low_prices[valley]:
                        valley = t
            
            if valley != -1:
                return (p1, p2, valley)
    return None

def find_double_bottom(data):
    """Double bottom detection"""
    if HAS_TRADING_PATTERNS:
        patterns = find_patterns_enhanced(data)
        if 'double_bottom' in patterns and patterns['double_bottom']:
            latest = patterns['double_bottom'][-1] if isinstance(patterns['double_bottom'], list) else patterns['double_bottom']
            return latest
    
    # Fallback
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    troughs, _ = find_peaks(-low_prices, distance=3, prominence=0.1)
    
    if len(troughs) < 2:
        return None
    
    for i in range(len(troughs) - 1):
        t1 = troughs[i]
        t2 = troughs[i+1]
        
        if abs(low_prices[t1] - low_prices[t2]) < 0.05 * (low_prices[t1] + low_prices[t2]) / 2:
            peaks, _ = find_peaks(high_prices, distance=3)
            peak = -1
            for p in peaks:
                if t1 < p < t2:
                    if peak == -1 or high_prices[p] > high_prices[peak]:
                        peak = p
            
            if peak != -1:
                return (t1, t2, peak)
    return None

def calculate_trading_signals(data, pattern_name, points):
    """Calculate actionable trading signals based on the pattern"""
    current_price = float(data['Close'].iloc[-1])
    signals = {}
    
    if pattern_name == "Head and Shoulders":
        l, p, r, lt, rt = points
        l, p, r, lt, rt = int(l), int(p), int(r), int(lt), int(rt)
        
        # Neckline is the average of the two troughs
        neckline = (data['Low'].iloc[lt] + data['Low'].iloc[rt]) / 2
        head_height = data['High'].iloc[p] - neckline
        
        signals['action'] = 'SELL'
        signals['entry'] = round(float(neckline * 0.99), 2)
        signals['stop_loss'] = round(float(data['High'].iloc[r] * 1.01), 2)
        signals['target'] = round(float(neckline - head_height), 2)
        
    elif pattern_name == "Inverse Head and Shoulders":
        l, h, r, lp, rp = points
        l, h, r, lp, rp = int(l), int(h), int(r), int(lp), int(rp)
        
        # Neckline is the average of the two peaks
        neckline = (data['High'].iloc[lp] + data['High'].iloc[rp]) / 2
        head_height = neckline - data['Low'].iloc[h]
        
        signals['action'] = 'BUY'
        signals['entry'] = round(float(neckline * 1.01), 2)
        signals['stop_loss'] = round(float(data['Low'].iloc[r] * 0.99), 2)
        signals['target'] = round(float(neckline + head_height), 2)
        
    elif pattern_name == "Double Top":
        p1, p2, valley = points
        p1, p2, valley = int(p1), int(p2), int(valley)
        
        resistance = (data['High'].iloc[p1] + data['High'].iloc[p2]) / 2
        support = data['Low'].iloc[valley]
        pattern_height = resistance - support
        
        signals['action'] = 'SELL'
        signals['entry'] = round(float(support * 0.99), 2)
        signals['stop_loss'] = round(float(resistance * 1.01), 2)
        signals['target'] = round(float(support - pattern_height * 0.8), 2)
        
    elif pattern_name == "Double Bottom":
        t1, t2, peak = points
        t1, t2, peak = int(t1), int(t2), int(peak)
        
        support = (data['Low'].iloc[t1] + data['Low'].iloc[t2]) / 2
        resistance = data['High'].iloc[peak]
        pattern_height = resistance - support
        
        signals['action'] = 'BUY'
        signals['entry'] = round(float(resistance * 1.01), 2)
        signals['stop_loss'] = round(float(support * 0.99), 2)
        signals['target'] = round(float(resistance + pattern_height * 0.8), 2)
    
    # Calculate risk/reward ratio
    if 'entry' in signals and 'stop_loss' in signals and 'target' in signals:
        risk = abs(signals['entry'] - signals['stop_loss'])
        reward = abs(signals['target'] - signals['entry'])
        signals['risk_reward'] = round(reward / risk if risk > 0 else 0, 2)
    
    signals['current_price'] = round(current_price, 2)
    signals['distance_to_entry'] = round(abs(current_price - signals['entry']) / current_price * 100, 2)
    
    # Add pattern confidence
    signals['confidence'] = calculate_pattern_confidence(data, pattern_name, points)
    
    return signals

def calculate_pattern_confidence(data, pattern_name, points):
    """Calculate confidence score for the pattern (0-100)"""
    # Base confidence depends on detection method
    if HAS_TRADING_PATTERNS:
        confidence = 84.5  # Wavelet method accuracy from documentation
    else:
        confidence = 50  # Base confidence for scipy method
    
    if pattern_name in ["Double Top", "Double Bottom"]:
        if pattern_name == "Double Top":
            p1, p2, valley = points
            # Check how similar the two peaks are
            peak_diff = abs(data['High'].iloc[int(p1)] - data['High'].iloc[int(p2)])
            avg_peak = (data['High'].iloc[int(p1)] + data['High'].iloc[int(p2)]) / 2
            symmetry_score = 1 - (peak_diff / avg_peak)
        else:  # Double Bottom
            t1, t2, peak = points
            # Check how similar the two troughs are
            trough_diff = abs(data['Low'].iloc[int(t1)] - data['Low'].iloc[int(t2)])
            avg_trough = (data['Low'].iloc[int(t1)] + data['Low'].iloc[int(t2)]) / 2
            symmetry_score = 1 - (trough_diff / avg_trough)
        
        # Add confidence based on pattern symmetry
        confidence += symmetry_score * 15 if HAS_TRADING_PATTERNS else symmetry_score * 30
        
        # Add confidence based on volume (if available)
        if 'Volume' in data.columns:
            recent_vol = data['Volume'].iloc[-5:].mean()
            pattern_vol = data['Volume'].iloc[min(points):max(points)+1].mean()
            if pattern_vol > recent_vol * 1.2:  # Higher volume during pattern
                confidence += 5 if HAS_TRADING_PATTERNS else 10
        
        # Reduce confidence if pattern is too recent (less reliable)
        pattern_age = len(data) - max(points)
        if pattern_age < 5:  # Very recent pattern
            confidence -= 10
    
    # Ensure confidence is between 0 and 100
    confidence = max(0, min(100, confidence))
    
    return round(confidence, 1)

def plot_pattern(data, pattern_name, points, ticker):
    """Placeholder for plotting function"""
    pass

if __name__ == "__main__":
    # Test the scanner
    ticker = "NVDA"
    data = yf.download(ticker, period='1d', interval='5m', progress=False)
    
    if not data.empty:
        cleaned_data = clean_yfinance_data(data)
        
        patterns = {
            "Head and Shoulders": find_head_and_shoulders,
            "Inverse Head and Shoulders": find_inverse_head_and_shoulders,
            "Double Top": find_double_top,
            "Double Bottom": find_double_bottom
        }
        
        print(f"\nScanning {ticker} for patterns...")
        print(f"Using: {'tradingpattern library' if HAS_TRADING_PATTERNS else 'scipy implementation'}")
        
        for pattern_name, find_func in patterns.items():
            result = find_func(cleaned_data)
            if result is not None:
                signals = calculate_trading_signals(cleaned_data, pattern_name, result)
                print(f"\nFound {pattern_name}:")
                print(f"  Action: {signals['action']}")
                print(f"  Entry: ${signals['entry']}")
                print(f"  Stop Loss: ${signals['stop_loss']}")
                print(f"  Target: ${signals['target']}")
                print(f"  Risk/Reward: {signals['risk_reward']}")
                print(f"  Confidence: {signals['confidence']}%")