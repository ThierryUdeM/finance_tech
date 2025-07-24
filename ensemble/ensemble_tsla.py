#!/usr/bin/env python3
"""
TSLA Ensemble Model
Combines momentum shapematching and simple technical for TSLA
Optimized for high-volatility EV stock with strong momentum characteristics
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .ensemble_base import EnsembleModel
from .momentum_shapematching.v1_TSLA import v1_tsla_model
from .simple_technical.simple_technical_tsla import simple_technical_tsla


def ensemble_tsla(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    TSLA-specific ensemble model
    
    Customizations:
    - Higher volatility tolerance
    - Aggressive momentum capture
    - Quick position changes
    """
    
    # Initialize ensemble with TSLA-specific parameters
    ensemble = EnsembleModel(
        momentum_model=v1_tsla_model,
        technical_model=simple_technical_tsla,
        ticker='TSLA',
        # Regime parameters - TSLA is volatile
        adx_threshold=28,  # Slightly lower than NVDA
        hurst_threshold=0.35,  # Even lower for quick momentum
        keltner_threshold=0.3,  # Higher for extreme volatility
        regime_smoothing=2,  # Less smoothing for quick changes
        # Position sizing - aggressive but aware of volatility
        full_size=0.9,  # Slightly less than full due to volatility
        half_size=0.5,
        quarter_size=0.25,
        min_confidence=0.5,  # Lower threshold for more trades
        # Signal thresholds
        long_threshold=0.5,  # Very sensitive
        short_threshold=0.5
    )
    
    # Calculate ensemble signals
    signals = ensemble.calculate_signals(train_data, test_data)
    
    # Post-processing for TSLA
    # During earnings or high news periods, be more conservative
    if 'volume' in test_data.columns:
        # Detect unusual volume spikes (potential news)
        vol_zscore = (test_data['volume'] - test_data['volume'].rolling(50).mean()) / test_data['volume'].rolling(50).std()
        high_vol_mask = vol_zscore > 3
        signals.loc[high_vol_mask, 'position_size'] *= 0.7
    
    # Apply minimum position size filter
    signals.loc[signals['position_size'] < 0.15, 'signal'] = 0
    signals.loc[signals['position_size'] < 0.15, 'position_size'] = 0
    
    # Store attribution for analysis
    if kwargs.get('return_attribution', False):
        return signals, ensemble.get_attribution_report()
    
    return signals[['signal']]  # Return only signal column for compatibility


# Standalone function for direct testing
def test_ensemble_tsla():
    """Test function to verify ensemble model works"""
    print("Testing TSLA Ensemble Model...")
    
    # Create dummy data
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    dummy_data = pd.DataFrame({
        'open': 200 + np.random.randn(1000).cumsum() * 2,
        'high': 202 + np.random.randn(1000).cumsum() * 2,
        'low': 198 + np.random.randn(1000).cumsum() * 2,
        'close': 200 + np.random.randn(1000).cumsum() * 2,
        'volume': np.random.randint(2000000, 10000000, 1000)
    }, index=dates)
    
    train = dummy_data[:800]
    test = dummy_data[800:]
    
    try:
        signals = ensemble_tsla(train, test)
        print(f"Success! Generated {len(signals)} signals")
        print(f"Signal distribution: {signals['signal'].value_counts()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ensemble_tsla()