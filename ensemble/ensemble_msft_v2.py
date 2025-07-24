#!/usr/bin/env python3
"""
MSFT V2 Ensemble Model
Using improved momentum and technical models with V2 base class
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_base_v2 import EnsembleModelV2
from momentum_shapematching.msft_improved import msft_improved_model
from simple_technical.simple_technical_msft import simple_technical_msft
from regime_weights import msft_regime_weights


def ensemble_msft_v2(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """MSFT V2 ensemble model using improved base class and improved models"""
    
    ensemble = EnsembleModelV2(
        momentum_model=msft_improved_model,
        technical_model=simple_technical_msft,
        ticker='MSFT',
        regime_weight_func=msft_regime_weights,
        # Conservative parameters for MSFT
        adx_threshold=22,
        hurst_threshold=0.5,
        keltner_threshold=0.12,
        regime_smoothing=15,
        # Position sizing
        full_size=0.7,
        half_size=0.35,
        quarter_size=0.15,
        min_confidence=0.7,
        # Signal thresholds
        long_threshold=0.65,
        short_threshold=0.35,
        # Risk management
        max_daily_trades=1,
        stop_loss_pct=0.015,
        trailing_stop_pct=0.01,
        # Probability calibration
        prob_calibration_strength=2.5
    )
    
    signals = ensemble.calculate_signals(train_data, test_data)
    
    return signals[['signal']]