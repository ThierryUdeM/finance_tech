#!/usr/bin/env python3
"""
Test just the indicator calculations without Azure or yfinance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SimpleTechnicalTester:
    def __init__(self):
        self.params = {
            'sma_fast': 10,
            'sma_slow': 30,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            'volume_sma': 20,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2
        }
    
    def create_test_data(self, n=100):
        """Create synthetic test data"""
        dates = pd.date_range(end=datetime.now(), periods=n, freq='15min')
        
        # Create realistic price movement
        base_price = 100
        prices = []
        for i in range(n):
            change = np.random.normal(0, 0.5)
            base_price *= (1 + change/100)
            prices.append(base_price)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * np.random.uniform(1.001, 1.01) for p in prices],
            'Low': [p * np.random.uniform(0.99, 0.999) for p in prices],
            'Close': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'Volume': np.random.randint(1000000, 5000000, n)
        }, index=dates)
        
        return df
    
    def calculate_indicators(self, data):
        """Calculate all technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_fast'] = df['Close'].rolling(window=self.params['sma_fast']).mean()
        df['SMA_slow'] = df['Close'].rolling(window=self.params['sma_slow']).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'], self.params['rsi_period'])
        
        # Stochastic
        df['STOCH_K'], df['STOCH_D'] = self.calculate_stochastic(
            df['High'], df['Low'], df['Close'],
            self.params['stoch_k_period'], self.params['stoch_d_period']
        )
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=self.params['bb_period']).mean()
        bb_std = df['Close'].rolling(window=self.params['bb_period']).std()
        
        # This is where the error was - ensure bb_std is a Series, not DataFrame
        print(f"bb_std type: {type(bb_std)}")
        print(f"bb_std shape: {bb_std.shape if hasattr(bb_std, 'shape') else 'N/A'}")
        
        df['BB_upper'] = df['BB_middle'] + (bb_std * self.params['bb_std'])
        df['BB_lower'] = df['BB_middle'] - (bb_std * self.params['bb_std'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent


def main():
    """Test the indicator calculations"""
    print("Testing indicator calculations...")
    
    tester = SimpleTechnicalTester()
    
    # Create test data
    print("\nCreating test data...")
    data = tester.create_test_data()
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {data.columns.tolist()}")
    
    # Test without multi-level columns (normal case)
    print("\n\nTest 1: Normal single-level columns")
    try:
        df = tester.calculate_indicators(data)
        print("✓ Success! Indicators calculated without error")
        print(f"Final dataframe has {len(df.columns)} columns")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test with multi-level columns (simulating yfinance output)
    print("\n\nTest 2: Multi-level columns (like yfinance)")
    data_multi = data.copy()
    # Create multi-level columns
    data_multi.columns = pd.MultiIndex.from_product([data.columns, ['NVDA']])
    print(f"Multi-level columns: {data_multi.columns.tolist()}")
    
    try:
        df = tester.calculate_indicators(data_multi)
        print("✗ This should have failed with multi-level columns!")
    except Exception as e:
        print(f"✓ Expected error with multi-level columns: {e}")
    
    # Test with flattened columns (the fix)
    print("\n\nTest 3: Flattened columns (after fix)")
    if isinstance(data_multi.columns, pd.MultiIndex):
        data_multi.columns = data_multi.columns.droplevel(1)
    print(f"Flattened columns: {data_multi.columns.tolist()}")
    
    try:
        df = tester.calculate_indicators(data_multi)
        print("✓ Success! Fix works - indicators calculated after flattening columns")
    except Exception as e:
        print(f"✗ Error even after fix: {e}")


if __name__ == "__main__":
    main()