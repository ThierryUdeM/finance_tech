#!/usr/bin/env python3
"""
NVDA Ensemble Model
Combines momentum shapematching and simple technical for NVDA
Optimized for high-momentum, volatile tech stock
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_momentum_technical.ensemble_base import EnsembleModel
from momentum_shapematching.nvda_v1 import momentum_shape_model
from simple_technical.simple_technical_nvda import simple_technical_nvda


def ensemble_nvda(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    NVDA-specific ensemble model
    
    Customizations:
    - Higher ADX threshold (30) for stronger trend detection
    - Lower Hurst threshold (0.4) to catch momentum earlier
    - Aggressive position sizing in strong trends
    """
    
    # Initialize ensemble with NVDA-specific parameters
    ensemble = EnsembleModel(
        momentum_model=momentum_shape_model,
        technical_model=simple_technical_nvda,
        ticker='NVDA',
        # Regime parameters - NVDA trends strongly
        adx_threshold=30,  # Higher threshold for NVDA's strong trends
        hurst_threshold=0.4,  # Lower to detect trends earlier
        keltner_threshold=0.25,  # Higher for NVDA's volatility
        regime_smoothing=3,
        # Position sizing - more aggressive for NVDA
        full_size=1.0,
        half_size=0.6,  # Slightly larger half position
        quarter_size=0.3,
        min_confidence=0.55,  # Slightly lower for more signals
        # Signal thresholds
        long_threshold=0.55,  # More sensitive
        short_threshold=0.45
    )
    
    # Calculate ensemble signals
    signals = ensemble.calculate_signals(train_data, test_data)
    
    # Post-processing for NVDA
    # During extreme volatility, reduce position sizes
    if 'volatility' in test_data.columns:
        high_vol_mask = test_data['volatility'] > test_data['volatility'].rolling(20).mean() * 1.5
        signals.loc[high_vol_mask, 'position_size'] *= 0.8
    
    # Apply minimum position size filter
    signals.loc[signals['position_size'] < 0.2, 'signal'] = 0
    signals.loc[signals['position_size'] < 0.2, 'position_size'] = 0
    
    # Store attribution for analysis
    if kwargs.get('return_attribution', False):
        return signals, ensemble.get_attribution_report()
    
    return signals[['signal']]  # Return only signal column for compatibility


# Standalone function for direct testing
def test_ensemble_nvda():
    """Test function to verify ensemble model works"""
    print("Testing NVDA Ensemble Model...")
    
    # Create dummy data
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2023-01-01', periods=1000, freq='15min')
    dummy_data = pd.DataFrame({
        'open': 100 + np.random.randn(1000).cumsum(),
        'high': 101 + np.random.randn(1000).cumsum(),
        'low': 99 + np.random.randn(1000).cumsum(),
        'close': 100 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(1000000, 5000000, 1000)
    }, index=dates)
    
    train = dummy_data[:800]
    test = dummy_data[800:]
    
    try:
        signals = ensemble_nvda(train, test)
        print(f"Success! Generated {len(signals)} signals")
        print(f"Signal distribution: {signals['signal'].value_counts()}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_ensemble_nvda()