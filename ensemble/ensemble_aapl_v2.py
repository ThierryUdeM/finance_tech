#!/usr/bin/env python3
"""
AAPL V2 Ensemble Model
Using improved momentum and technical models with V2 base class
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ensemble_base_v2 import EnsembleModelV2
from momentum_shapematching.aapl_improved import aapl_improved_model
from simple_technical.simple_technical_aapl import simple_technical_aapl
from regime_weights import aapl_regime_weights


def ensemble_aapl_v2(train_data: pd.DataFrame, test_data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """AAPL V2 ensemble model using improved base class and improved models"""
    
    ensemble = EnsembleModelV2(
        momentum_model=aapl_improved_model,
        technical_model=simple_technical_aapl,
        ticker='AAPL',
        regime_weight_func=aapl_regime_weights,
        # Conservative parameters for AAPL
        adx_threshold=25,
        hurst_threshold=0.45,
        keltner_threshold=0.15,
        regime_smoothing=10,
        # Position sizing
        full_size=0.8,
        half_size=0.4,
        quarter_size=0.2,
        min_confidence=0.65,
        # Signal thresholds
        long_threshold=0.6,
        short_threshold=0.4,
        # Risk management
        max_daily_trades=2,
        stop_loss_pct=0.02,
        trailing_stop_pct=0.015,
        # Probability calibration
        prob_calibration_strength=2.0
    )
    
    signals = ensemble.calculate_signals(train_data, test_data)
    
    return signals[['signal']]