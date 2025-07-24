#!/usr/bin/env python3
"""
TSLA Ensemble Model
Combines momentum shapematching and simple technical for TSLA
Optimized for extreme volatility and news-driven moves
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_momentum_technical.ensemble_base import EnsembleModel
from momentum_shapematching.v1_TSLA import v1_tsla_model
from simple_technical.simple_technical_tsla import simple_technical_tsla


def ensemble_tsla(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    TSLA-specific ensemble model
    
    Customizations:
    - Higher ADX threshold (28) for strong momentum moves
    - Lower Hurst threshold (0.42) for early trend detection
    - Adaptive position sizing based on volatility
    """
    
    # Initialize ensemble with TSLA-specific parameters
    ensemble = EnsembleModel(
        momentum_model=v1_tsla_model,
        technical_model=simple_technical_tsla,
        ticker='TSLA',
        # Regime parameters - TSLA has extreme moves
        adx_threshold=28,  # Higher for strong trends
        hurst_threshold=0.42,  # Lower for momentum
        keltner_threshold=0.3,  # Higher for volatility
        regime_smoothing=2,  # Less smoothing for fast reaction
        # Position sizing - adaptive for TSLA
        full_size=0.8,  # Reduced full size due to volatility
        half_size=0.4,  
        quarter_size=0.2,
        min_confidence=0.6,
        # Signal thresholds
        long_threshold=0.58,  # Slightly tighter
        short_threshold=0.42
    )
    
    # Calculate ensemble signals
    signals = ensemble.calculate_signals(train_data, test_data)
    
    # Post-processing for TSLA
    # Calculate rolling volatility for position sizing
    if 'close' in test_data.columns:
        returns = test_data['close'].pct_change()
        rolling_vol = returns.rolling(20).std()
        vol_percentile = rolling_vol.rank(pct=True)
        
        # Reduce position size in extreme volatility
        extreme_vol = vol_percentile > 0.9
        signals.loc[extreme_vol, 'position_size'] *= 0.7
        
        # Increase position size in low volatility (rare for TSLA)
        low_vol = vol_percentile < 0.2
        signals.loc[low_vol, 'position_size'] *= 1.3
        
        # Cap position sizes
        signals['position_size'] = signals['position_size'].clip(upper=0.8)
    
    # TSLA-specific: Skip signals around known event times if possible
    # (would need event calendar integration)
    
    # Apply minimum position size filter
    signals.loc[signals['position_size'] < 0.15, 'signal'] = 0
    signals.loc[signals['position_size'] < 0.15, 'position_size'] = 0
    
    # Limit daily signals for TSLA (it overtraded in tests)
    daily_count = signals.groupby(signals.index.date)['signal'].transform(
        lambda x: (x != 0).cumsum()
    )
    signals.loc[daily_count > 3, 'signal'] = 0  # Max 3 signals per day
    signals.loc[daily_count > 3, 'position_size'] = 0
    
    # Store attribution for analysis
    if kwargs.get('return_attribution', False):
        return signals, ensemble.get_attribution_report()
    
    return signals[['signal']]  # Return only signal column for compatibility