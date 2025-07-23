#!/usr/bin/env python3
"""
Test the optimized scanner without Azure dependencies
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Simplified version of the optimized scanner for testing
class TestOptimizedScanner:
    def __init__(self):
        # Optimized parameters
        self.params = {
            'sma_fast': 8,
            'sma_slow': 24,
            'rsi_period': 12,
            'stoch_k_period': 12,
            'stoch_d_period': 3,
            'volume_sma': 16,
            'atr_period': 12,
            'bb_period': 18,
            'bb_std': 2.1,
            'macd_fast': 10,
            'macd_slow': 24,
            'macd_signal': 8
        }
        
        # ML-optimized weights
        self.weights = {
            'ma_cross': 1.8,
            'rsi': 2.2,
            'stoch': 1.2,
            'bb': 1.4,
            'macd': 1.8,
            'volume': 0.8,
            'momentum': 1.6,
            'volatility': 1.0
        }
        
        self.buy_threshold = 5.2
        self.sell_threshold = -5.2
    
    def calculate_rsi_ema(self, prices, period=14):
        """Enhanced RSI with EMA smoothing"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def test_scan(self, ticker='NVDA'):
        """Test the optimized scanning logic"""
        print(f"Testing optimized scanner on {ticker}")
        
        # Get data
        data = yf.download(ticker, period='5d', interval='15m', progress=False)
        
        if data.empty:
            print("No data retrieved")
            return None
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Calculate some key indicators
        data['RSI'] = self.calculate_rsi_ema(data['Close'], self.params['rsi_period'])
        data['SMA_fast'] = data['Close'].rolling(window=self.params['sma_fast']).mean()
        data['SMA_slow'] = data['Close'].rolling(window=self.params['sma_slow']).mean()
        data['Volume_EMA'] = data['Volume'].ewm(span=self.params['volume_sma']).mean()
        data['Volume_ratio'] = data['Volume'] / data['Volume_EMA']
        
        # Get current values
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_volume_ratio = data['Volume_ratio'].iloc[-1]
        
        # Simple signal logic test
        signal_score = 0
        
        # RSI component
        if current_rsi < 32:
            signal_score += self.weights['rsi']
        elif current_rsi > 68:
            signal_score -= self.weights['rsi']
        
        # MA component
        if data['SMA_fast'].iloc[-1] > data['SMA_slow'].iloc[-1]:
            signal_score += self.weights['ma_cross']
        else:
            signal_score -= self.weights['ma_cross']
        
        # Volume component
        if current_volume_ratio > 1.3:
            signal_score += self.weights['volume']
        
        # Determine signal
        if signal_score >= self.buy_threshold:
            signal = 'BUY'
            confidence = min(signal_score / 10, 0.95)
        elif signal_score <= self.sell_threshold:
            signal = 'SELL'
            confidence = min(abs(signal_score) / 10, 0.95)
        else:
            signal = 'HOLD'
            confidence = 0.5
        
        print(f"Results for {ticker}:")
        print(f"  Price: ${current_price:.2f}")
        print(f"  RSI: {current_rsi:.1f}")
        print(f"  Volume Ratio: {current_volume_ratio:.2f}")
        print(f"  Signal Score: {signal_score:.1f}")
        print(f"  Signal: {signal} ({confidence:.1%} confidence)")
        print(f"  Thresholds: Buy >= {self.buy_threshold}, Sell <= {self.sell_threshold}")
        
        return {
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'price': current_price,
            'signal_score': signal_score
        }

if __name__ == "__main__":
    scanner = TestOptimizedScanner()
    
    # Test on NVDA
    result = scanner.test_scan('NVDA')
    
    if result:
        print(f"\nOptimized scanner test completed successfully!")
        print(f"Generated {result['signal']} signal with {result['confidence']:.1%} confidence")
    else:
        print("Test failed")