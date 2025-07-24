"""
Hybrid Momentum-Technical Ensemble Models
Combines momentum shapematching and simple technical models based on market regime
"""

from .market_regime import classify_regime, get_regime_weights
from .ensemble_base import EnsembleModel

__all__ = ['classify_regime', 'get_regime_weights', 'EnsembleModel']