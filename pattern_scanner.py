import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
# import mplfinance as mpf  # Commented out for GitHub Actions
import re # Import the re module for regular expressions
import ast # Import ast for literal_eval
import os
from datetime import datetime

def clean_yfinance_data(data):
    """
    Cleans data from yfinance, handling potential multi-index columns
    and ensuring data is numeric.
    """
    print("--- Inside clean_yfinance_data ---")
    print("Columns received:", data.columns)
    print("Head of data received:\n", data.head())

    # Handle column names that are string representations of tuples from R
    new_columns = []
    for col in data.columns:
        try:
            # Attempt to evaluate as a literal tuple
            col_tuple = ast.literal_eval(str(col)) # Ensure it's a string before eval
            if isinstance(col_tuple, tuple) and len(col_tuple) > 0:
                new_columns.append(col_tuple[0]) # Take the first element (e.g., 'Open')
            else:
                new_columns.append(col) # Keep as is if not a tuple
        except (ValueError, SyntaxError):
            # Fallback if literal_eval fails (e.g., not a tuple string)
            # Try a simple string split if it looks like "('Open', 'NVDA')"
            if isinstance(col, str) and col.startswith("('") and ", '" in col and col.endswith("')"):
                new_col = col.split(",")[0].replace("('", "").replace("'", "")
                new_columns.append(new_col)
            else:
                new_columns.append(col) # Keep as is
    data.columns = new_columns

    print("Columns after parsing attempt:", data.columns)

    # Ensure required columns are numeric, coercing errors
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            raise ValueError(f"Required column '{col}' not found in data.")

    data.dropna(inplace=True)

    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    print("--- Exiting clean_yfinance_data ---")
    return data

def find_head_and_shoulders(data):
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    
    peaks, _ = find_peaks(high_prices, distance=3, width=2)  # Reduced for intraday

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
            troughs, _ = find_peaks(-low_prices, distance=3, width=2)  # Reduced for intraday
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
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    troughs, _ = find_peaks(-low_prices, distance=3, width=2)  # Reduced for intraday

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
            peaks, _ = find_peaks(high_prices, distance=3, width=2)  # Reduced for intraday
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
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    peaks, _ = find_peaks(high_prices, distance=3, prominence=0.1)  # Reduced for intraday

    if len(peaks) < 2:
        return None

    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i+1]

        if abs(high_prices[p1] - high_prices[p2]) < 0.05 * (high_prices[p1] + high_prices[p2]) / 2:
            troughs, _ = find_peaks(-low_prices, distance=3)  # Reduced for intraday
            valley = -1
            for t in troughs:
                if p1 < t < p2:
                    if valley == -1 or low_prices[t] < low_prices[valley]:
                        valley = t

            if valley != -1:
                return (p1, p2, valley)
    return None

def find_double_bottom(data):
    if data.empty:
        return None
    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()
    troughs, _ = find_peaks(-low_prices, distance=3, prominence=0.1)  # Reduced for intraday

    if len(troughs) < 2:
        return None

    for i in range(len(troughs) - 1):
        t1 = troughs[i]
        t2 = troughs[i+1]

        if abs(low_prices[t1] - low_prices[t2]) < 0.05 * (low_prices[t1] + low_prices[t2]) / 2:
            peaks, _ = find_peaks(high_prices, distance=3)  # Reduced for intraday
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
    current_price = data['Close'].iloc[-1]
    signals = {}
    
    if pattern_name == "Head and Shoulders":
        l, p, r, lt, rt = points
        # Neckline is the average of the two troughs
        neckline = (data['Low'].iloc[int(lt)] + data['Low'].iloc[int(rt)]) / 2
        head_height = data['High'].iloc[int(p)] - neckline
        
        signals['action'] = 'SELL'
        signals['entry'] = round(neckline * 0.99, 2)  # Enter on break below neckline
        signals['stop_loss'] = round(data['High'].iloc[int(r)] * 1.01, 2)  # Above right shoulder
        signals['target'] = round(neckline - head_height, 2)  # Project head height below neckline
        signals['risk_reward'] = round(abs(signals['entry'] - signals['target']) / abs(signals['stop_loss'] - signals['entry']), 2)
        
    elif pattern_name == "Inverse Head and Shoulders":
        l, h, r, lp, rp = points
        # Neckline is the average of the two peaks
        neckline = (data['High'].iloc[int(lp)] + data['High'].iloc[int(rp)]) / 2
        head_height = neckline - data['Low'].iloc[int(h)]
        
        signals['action'] = 'BUY'
        signals['entry'] = round(neckline * 1.01, 2)  # Enter on break above neckline
        signals['stop_loss'] = round(data['Low'].iloc[int(r)] * 0.99, 2)  # Below right shoulder
        signals['target'] = round(neckline + head_height, 2)  # Project head height above neckline
        signals['risk_reward'] = round(abs(signals['target'] - signals['entry']) / abs(signals['entry'] - signals['stop_loss']), 2)
        
    elif pattern_name == "Double Top":
        p1, p2, valley = points
        resistance = (data['High'].iloc[int(p1)] + data['High'].iloc[int(p2)]) / 2
        support = data['Low'].iloc[int(valley)]
        pattern_height = resistance - support
        
        signals['action'] = 'SELL'
        signals['entry'] = round(support * 0.99, 2)  # Enter on break below valley
        signals['stop_loss'] = round(resistance * 1.01, 2)  # Above the double top
        signals['target'] = round(support - pattern_height * 0.8, 2)  # Project 80% of pattern height below support
        signals['risk_reward'] = round(abs(signals['entry'] - signals['target']) / abs(signals['stop_loss'] - signals['entry']), 2)
        
    elif pattern_name == "Double Bottom":
        t1, t2, peak = points
        support = (data['Low'].iloc[int(t1)] + data['Low'].iloc[int(t2)]) / 2
        resistance = data['High'].iloc[int(peak)]
        pattern_height = resistance - support
        
        signals['action'] = 'BUY'
        signals['entry'] = round(resistance * 1.01, 2)  # Enter on break above peak
        signals['stop_loss'] = round(support * 0.99, 2)  # Below the double bottom
        signals['target'] = round(resistance + pattern_height * 0.8, 2)  # Project 80% of pattern height above resistance
        signals['risk_reward'] = round(abs(signals['target'] - signals['entry']) / abs(signals['entry'] - signals['stop_loss']), 2)
    
    signals['current_price'] = round(current_price, 2)
    signals['distance_to_entry'] = round(abs(current_price - signals['entry']) / current_price * 100, 2)
    
    return signals

def plot_pattern(data, pattern_name, points, ticker):
    # Plotting disabled for GitHub Actions
    return
    if points is None:
        print(f"No {pattern_name} pattern found.")
        return

    # Ensure the index is a DatetimeIndex for mplfinance
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Print pattern details
    print(f"\n=== {pattern_name} Pattern Details ===")
    if pattern_name == "Head and Shoulders":
        l, p, r, lt, rt = points
        print(f"Left Shoulder: {data.index[int(l)]} - Price: ${data['High'].iloc[int(l)]:.2f}")
        print(f"Head: {data.index[int(p)]} - Price: ${data['High'].iloc[int(p)]:.2f}")
        print(f"Right Shoulder: {data.index[int(r)]} - Price: ${data['High'].iloc[int(r)]:.2f}")
        print(f"Pattern suggests bearish reversal")
    elif pattern_name == "Inverse Head and Shoulders":
        l, h, r, lp, rp = points
        print(f"Left Shoulder: {data.index[int(l)]} - Price: ${data['Low'].iloc[int(l)]:.2f}")
        print(f"Head: {data.index[int(h)]} - Price: ${data['Low'].iloc[int(h)]:.2f}")
        print(f"Right Shoulder: {data.index[int(r)]} - Price: ${data['Low'].iloc[int(r)]:.2f}")
        print(f"Pattern suggests bullish reversal")
    elif pattern_name == "Double Top":
        p1, p2, valley = points
        print(f"First Top: {data.index[int(p1)]} - Price: ${data['High'].iloc[int(p1)]:.2f}")
        print(f"Second Top: {data.index[int(p2)]} - Price: ${data['High'].iloc[int(p2)]:.2f}")
        print(f"Valley: {data.index[int(valley)]} - Price: ${data['Low'].iloc[int(valley)]:.2f}")
        print(f"Pattern suggests bearish reversal")
    elif pattern_name == "Double Bottom":
        t1, t2, peak = points
        print(f"First Bottom: {data.index[int(t1)]} - Price: ${data['Low'].iloc[int(t1)]:.2f}")
        print(f"Second Bottom: {data.index[int(t2)]} - Price: ${data['Low'].iloc[int(t2)]:.2f}")
        print(f"Peak: {data.index[int(peak)]} - Price: ${data['High'].iloc[int(peak)]:.2f}")
        print(f"Pattern suggests bullish reversal")
    
    # Calculate and print trading signals
    signals = calculate_trading_signals(data, pattern_name, points)
    print(f"\n>>> TRADING SIGNALS <<<")
    print(f"Current Price: ${signals['current_price']}")
    print(f"ACTION: {signals['action']}")
    print(f"Entry Price: ${signals['entry']} (on breakout)")
    print(f"Stop Loss: ${signals['stop_loss']}")
    print(f"Target Price: ${signals['target']}")
    print(f"Risk/Reward Ratio: 1:{signals['risk_reward']}")
    print(f"Distance to Entry: {signals['distance_to_entry']}%")
    
    if signals['current_price'] > signals['entry'] and signals['action'] == 'BUY':
        print("⚠️  Pattern already triggered - price above entry")
    elif signals['current_price'] < signals['entry'] and signals['action'] == 'SELL':
        print("⚠️  Pattern already triggered - price below entry")

    high_prices = data['High'].to_numpy().flatten()
    low_prices = data['Low'].to_numpy().flatten()

    lines = []
    if pattern_name == "Head and Shoulders":
        l, p, r, _, _ = points
        l_idx = int(np.round(l))
        lines.append((data.index[l_idx], high_prices[l_idx]))
        p_idx = int(np.round(p))
        lines.append((data.index[p_idx], high_prices[p_idx]))
        r_idx = int(np.round(r))
        lines.append((data.index[r_idx], high_prices[r_idx]))
    elif pattern_name == "Inverse Head and Shoulders":
        l, h, r, _, _ = points
        l_idx = int(np.round(l))
        lines.append((data.index[l_idx], low_prices[l_idx]))
        h_idx = int(np.round(h))
        lines.append((data.index[h_idx], low_prices[h_idx]))
        r_idx = int(np.round(r))
        lines.append((data.index[r_idx], low_prices[r_idx]))
    elif pattern_name == "Double Top":
        p1, p2, _ = points
        p1_idx = int(np.round(p1))
        lines.append((data.index[p1_idx], high_prices[p1_idx]))
        p2_idx = int(np.round(p2))
        lines.append((data.index[p2_idx], high_prices[p2_idx]))
    elif pattern_name == "Double Bottom":
        t1, t2, _ = points
        t1_idx = int(np.round(t1))
        lines.append((data.index[t1_idx], low_prices[t1_idx]))
        t2_idx = int(np.round(t2))
        lines.append((data.index[t2_idx], low_prices[t2_idx]))

    # Create output directory if it doesn't exist
    output_dir = "chart_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{ticker}_{pattern_name.replace(' ', '_')}_{timestamp}.png"
    
    # For better visualization, show only a window around the pattern
    if lines:
        # Get the indices of the pattern points
        pattern_indices = []
        if pattern_name in ["Head and Shoulders", "Inverse Head and Shoulders"]:
            l, h_or_p, r, _, _ = points
            pattern_indices = [int(l), int(h_or_p), int(r)]
        elif pattern_name == "Double Top":
            p1, p2, _ = points
            pattern_indices = [int(p1), int(p2)]
        elif pattern_name == "Double Bottom":
            t1, t2, _ = points
            pattern_indices = [int(t1), int(t2)]
        
        # Create a window around the pattern (20% buffer on each side)
        min_idx = max(0, min(pattern_indices) - int(0.2 * (max(pattern_indices) - min(pattern_indices))))
        max_idx = min(len(data) - 1, max(pattern_indices) + int(0.2 * (max(pattern_indices) - min(pattern_indices))))
        
        # If window is too small, expand it
        if max_idx - min_idx < 50:
            min_idx = max(0, min_idx - 25)
            max_idx = min(len(data) - 1, max_idx + 25)
        
        window_data = data.iloc[min_idx:max_idx+1]
        
        # Adjust line coordinates for the window
        window_lines = []
        for date, price in lines:
            if data.index[min_idx] <= date <= data.index[max_idx]:
                window_lines.append((date, price))
        
        alines = dict(alines=[window_lines], colors=['r'], linewidths=[2])
        mpf.plot(window_data, type='candle', style='yahoo',
                 title=f'{ticker} - {pattern_name} (Focused View)',
                 alines=alines,
                 volume=True,
                 warn_too_much_data=800,
                 savefig=dict(fname=filename, dpi=150, bbox_inches='tight'))
        print(f"Chart saved to: {filename}")
    else:
        mpf.plot(data, type='candle', style='yahoo',
                 title=f'{ticker} - No Pattern Found (Intraday)',
                 volume=True,
                 warn_too_much_data=800,
                 savefig=dict(fname=filename, dpi=150, bbox_inches='tight'))
        print(f"Chart saved to: {filename}")