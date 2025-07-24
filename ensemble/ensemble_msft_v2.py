#!/usr/bin/env python3
"""
Ensemble Model for MSFT V2
Improved version with better calibration and risk management
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.hybrid_momentum_technical.ensemble_base_v2 import EnsembleModelV2
from model.hybrid_momentum_technical.regime_weights import msft_regime_weights
from model.momentum_shapematching.msft_improved import msft_improved_model
from model.simple_technical.simple_technical_msft import simple_technical_msft

import pandas as pd


def ensemble_msft_v2(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    MSFT Ensemble Model V2
    - Conservative balanced approach
    - Daily trade limit of 2
    - Higher confidence thresholds
    """
    
    # Initialize ensemble with MSFT-specific parameters
    ensemble = EnsembleModelV2(
        momentum_model=msft_improved_model,
        technical_model=simple_technical_msft,
        ticker='MSFT',
        # Regime parameters
        adx_threshold=28,  # Slightly higher for clearer trends
        hurst_threshold=0.48,
        keltner_threshold=0.2,
        regime_smoothing=5,
        # Position sizing (very conservative)
        full_size=1.0,
        half_size=0.5,
        quarter_size=0.15,
        min_confidence=0.70,  # Slightly relaxed from 0.75
        # Signal thresholds (most selective)
        long_threshold=0.7,  # Very selective
        short_threshold=0.3,
        # Risk management
        max_daily_trades=2,  # New daily limit
        atr_stop_multiplier=0.5,
        trailing_start_r=0.5,
        portfolio_stop_r=2.0,
        # MSFT-specific regime weights (conservative balance)
        regime_weight_func=msft_regime_weights,
        # Probability calibration
        prob_calibration_strength=2.0  # Strongest calibration for clarity
    )
    
    # Generate signals
    signals = ensemble.calculate_signals(train_data, test_data)
    
    # Return only the signal column for compatibility
    return signals[['signal']]