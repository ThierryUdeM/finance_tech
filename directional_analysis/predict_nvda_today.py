#!/usr/bin/env python3
"""
Predict NVDA's direction for today using Databento data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from intraday_pattern_matcher_enhanced import forecast_shape
import warnings
warnings.filterwarnings('ignore')

def get_current_nvda_price():
    """Get the most recent NVDA price from our data"""
    df = pd.read_csv("NVDA_15min_pattern_ready.csv", index_col=0, parse_dates=True)
    latest_bar = df.iloc[-1]
    latest_time = df.index[-1]
    return latest_bar['Close'], latest_time

def analyze_nvda_patterns():
    """Run comprehensive pattern analysis for NVDA"""
    
    print("="*70)
    print("NVDA DIRECTION PREDICTION ANALYSIS")
    print("="*70)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get latest price
    latest_price, latest_time = get_current_nvda_price()
    print(f"\nLatest Data Point:")
    print(f"  Time: {latest_time}")
    print(f"  Price: ${latest_price:.2f}")
    
    # Test different pattern lengths (in 15-min bars)
    pattern_configs = [
        (8, 20, "2 hours"),      # 8 bars = 2 hours, K=20
        (12, 30, "3 hours"),     # 12 bars = 3 hours, K=30
        (20, 50, "5 hours"),     # 20 bars = 5 hours, K=50
        (28, 70, "7 hours"),     # 28 bars = 7 hours (full day), K=70
    ]
    
    print("\n" + "-"*70)
    print("PATTERN MATCHING RESULTS")
    print("-"*70)
    
    all_predictions = {
        '1h': [],
        '3h': [],
        'eod': []
    }
    
    for query_length, k_neighbors, description in pattern_configs:
        print(f"\nPattern Length: {description} ({query_length} bars)")
        
        try:
            result = forecast_shape(
                "NVDA", 
                interval="15m",
                query_length=query_length, 
                K=k_neighbors
            )
            
            print(f"  Patterns analyzed: {result['library_size']:,}")
            print(f"  Nearest neighbors used: {k_neighbors}")
            print(f"  Predictions:")
            print(f"    1-hour:  {result['1h']*100:+.3f}%")
            print(f"    3-hour:  {result['3h']*100:+.3f}%")
            print(f"    EOD:     {result['eod']*100:+.3f}%")
            
            # Collect predictions for averaging
            all_predictions['1h'].append(result['1h'])
            all_predictions['3h'].append(result['3h'])
            all_predictions['eod'].append(result['eod'])
            
        except Exception as e:
            print(f"  Error: {e}")
    
    # Calculate ensemble predictions
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTION (Average of all pattern lengths)")
    print("="*70)
    
    ensemble_1h = np.mean(all_predictions['1h']) * 100
    ensemble_3h = np.mean(all_predictions['3h']) * 100
    ensemble_eod = np.mean(all_predictions['eod']) * 100
    
    print(f"\n1-Hour Prediction: {ensemble_1h:+.3f}%")
    print(f"  Direction: {'BULLISH ↑' if ensemble_1h > 0.1 else 'BEARISH ↓' if ensemble_1h < -0.1 else 'NEUTRAL →'}")
    print(f"  Confidence: {'HIGH' if abs(ensemble_1h) > 0.5 else 'MEDIUM' if abs(ensemble_1h) > 0.2 else 'LOW'}")
    
    print(f"\n3-Hour Prediction: {ensemble_3h:+.3f}%")
    print(f"  Direction: {'BULLISH ↑' if ensemble_3h > 0.1 else 'BEARISH ↓' if ensemble_3h < -0.1 else 'NEUTRAL →'}")
    print(f"  Confidence: {'HIGH' if abs(ensemble_3h) > 0.5 else 'MEDIUM' if abs(ensemble_3h) > 0.2 else 'LOW'}")
    
    print(f"\nEnd-of-Day Prediction: {ensemble_eod:+.3f}%")
    print(f"  Direction: {'BULLISH ↑' if ensemble_eod > 0.1 else 'BEARISH ↓' if ensemble_eod < -0.1 else 'NEUTRAL →'}")
    print(f"  Confidence: {'HIGH' if abs(ensemble_eod) > 0.5 else 'MEDIUM' if abs(ensemble_eod) > 0.2 else 'LOW'}")
    
    # Price targets
    print("\n" + "-"*70)
    print("PRICE TARGETS")
    print("-"*70)
    
    target_1h = latest_price * (1 + ensemble_1h/100)
    target_3h = latest_price * (1 + ensemble_3h/100)
    target_eod = latest_price * (1 + ensemble_eod/100)
    
    print(f"\nCurrent Price: ${latest_price:.2f}")
    print(f"\n1-Hour Target: ${target_1h:.2f} ({target_1h - latest_price:+.2f})")
    print(f"3-Hour Target: ${target_3h:.2f} ({target_3h - latest_price:+.2f})")
    print(f"EOD Target:    ${target_eod:.2f} ({target_eod - latest_price:+.2f})")
    
    # Historical accuracy check
    print("\n" + "-"*70)
    print("PATTERN QUALITY METRICS")
    print("-"*70)
    
    # Load data to check pattern density
    df = pd.read_csv("NVDA_15min_pattern_ready.csv", index_col=0, parse_dates=True)
    total_bars = len(df)
    total_days = len(df.groupby(df.index.date))
    
    print(f"\nData Statistics:")
    print(f"  Total 15-min bars: {total_bars:,}")
    print(f"  Trading days: {total_days}")
    print(f"  Data range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"  Average patterns per config: {np.mean([r['library_size'] for r in [forecast_shape('NVDA', query_length=20, K=1)]]):,.0f}")
    
    # Volatility context
    recent_prices = df['Close'].iloc[-20:]  # Last 5 hours
    recent_volatility = (recent_prices.pct_change().std() * np.sqrt(252 * 26)) * 100  # Annualized
    
    print(f"\nMarket Context:")
    print(f"  Recent volatility (annualized): {recent_volatility:.1f}%")
    print(f"  Volatility level: {'HIGH' if recent_volatility > 60 else 'MEDIUM' if recent_volatility > 40 else 'LOW'}")
    
    # Summary recommendation
    print("\n" + "="*70)
    print("TRADING RECOMMENDATION")
    print("="*70)
    
    # Determine overall bias
    signals = [ensemble_1h, ensemble_3h, ensemble_eod]
    bullish_signals = sum(1 for s in signals if s > 0.1)
    bearish_signals = sum(1 for s in signals if s < -0.1)
    
    if bullish_signals > bearish_signals:
        overall_bias = "BULLISH"
        action = "Consider LONG positions"
    elif bearish_signals > bullish_signals:
        overall_bias = "BEARISH"
        action = "Consider SHORT positions"
    else:
        overall_bias = "NEUTRAL"
        action = "Stay flat or use tight stops"
    
    print(f"\nOverall Bias: {overall_bias}")
    print(f"Suggested Action: {action}")
    
    # Risk levels
    if abs(ensemble_1h) > 0.5:
        print(f"\nRisk Level: HIGH")
        print(f"  - Strong directional signal")
        print(f"  - Consider position sizing accordingly")
    else:
        print(f"\nRisk Level: MODERATE")
        print(f"  - Mixed or weak signals")
        print(f"  - Use smaller position sizes")
    
    print("\n" + "="*70)
    print("Disclaimer: These predictions are based on historical patterns.")
    print("Past performance does not guarantee future results.")
    print("Always use proper risk management.")
    print("="*70)

if __name__ == "__main__":
    analyze_nvda_patterns()