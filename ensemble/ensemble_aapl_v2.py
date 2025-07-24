#!/usr/bin/env python3
"""
Ensemble Model for AAPL V2
Improved version with better calibration and risk management
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.hybrid_momentum_technical.ensemble_base_v2 import EnsembleModelV2
from model.hybrid_momentum_technical.regime_weights import aapl_regime_weights
from model.momentum_shapematching.aapl_improved import aapl_improved_model
from model.simple_technical.simple_technical_aapl import simple_technical_aapl

import pandas as pd


def ensemble_aapl_v2(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    AAPL Ensemble Model V2
    - Heavy technical/mean-reversion focus
    - Conservative position sizing
    - Already has daily limit in base model (2/day)
    """
    
    # Initialize ensemble with AAPL-specific parameters
    ensemble = EnsembleModelV2(
        momentum_model=aapl_improved_model,
        technical_model=simple_technical_aapl,
        ticker='AAPL',
        # Regime parameters
        adx_threshold=25,
        hurst_threshold=0.5,  # Higher threshold - AAPL is often mean-reverting
        keltner_threshold=0.2,
        regime_smoothing=7,  # More smoothing for stable stock
        # Position sizing (conservative)
        full_size=1.0,
        half_size=0.5,
        quarter_size=0.15,
        min_confidence=0.65,  # Slightly relaxed from 0.7
        # Signal thresholds (wider for selectivity)
        long_threshold=0.7,  # More selective
        short_threshold=0.3,
        # Risk management
        max_daily_trades=2,  # Maintain existing limit
        atr_stop_multiplier=0.5,
        trailing_start_r=0.5,
        portfolio_stop_r=2.0,
        # AAPL-specific regime weights (minimal momentum)
        regime_weight_func=aapl_regime_weights,
        # Probability calibration (stronger separation)
        prob_calibration_strength=1.8
    )
    
    # Generate signals
    signals = ensemble.calculate_signals(train_data, test_data)
    
    # Return only the signal column for compatibility
    return signals[['signal']]