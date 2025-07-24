#!/usr/bin/env python3
"""
Ticker-specific regime weight functions
"""

from typing import Tuple


def nvda_regime_weights(regime_score: float) -> Tuple[float, float]:
    """
    NVDA: Strong momentum bias in trends
    """
    # NVDA is momentum-driven, give high weight to momentum in trends
    w_momentum = 0.5 + 0.45 * max(0, regime_score)  # 0.5 to 0.95 in strong trends
    w_technical = 1.0 - w_momentum
    return w_momentum, w_technical


def aapl_regime_weights(regime_score: float) -> Tuple[float, float]:
    """
    AAPL: Technical/mean-reversion focused, minimal momentum
    """
    # AAPL is range-bound, heavily favor technical indicators
    # Only use momentum in very strong trends
    w_momentum = max(0, regime_score * 0.3)  # 0 to 0.3 max
    w_technical = 1.0 - w_momentum
    return w_momentum, w_technical


def msft_regime_weights(regime_score: float) -> Tuple[float, float]:
    """
    MSFT: Balanced but conservative, avoid momentum in ranges
    """
    # MSFT needs careful balance, only momentum in clear trends
    if regime_score > 0.5:  # Strong trend
        w_momentum = 0.4 + 0.3 * (regime_score - 0.5)  # 0.4 to 0.55
    else:
        w_momentum = 0.1  # Minimal momentum in ranges
    w_technical = 1.0 - w_momentum
    return w_momentum, w_technical


def tsla_regime_weights(regime_score: float) -> Tuple[float, float]:
    """
    TSLA: Aggressive momentum in trends, but respect ranges
    """
    # TSLA has extreme moves, use momentum aggressively in trends
    if regime_score > 0.3:  # Trending
        w_momentum = 0.6 + 0.35 * (regime_score - 0.3) / 0.7  # 0.6 to 0.95
    elif regime_score < -0.3:  # Strong range
        w_momentum = 0.1  # Very low momentum
    else:  # Uncertain
        w_momentum = 0.3
    w_technical = 1.0 - w_momentum
    return w_momentum, w_technical