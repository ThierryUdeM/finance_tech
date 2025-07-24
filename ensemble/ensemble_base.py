#!/usr/bin/env python3
"""
Ensemble Model Base Class
Combines momentum shapematching and simple technical models based on market regime
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

from .market_regime import classify_regime, get_regime_weights, smooth_regime


class EnsembleModel:
    """
    Base class for ensemble models that combine momentum and technical strategies
    """
    
    def __init__(self, 
                 momentum_model: Callable,
                 technical_model: Callable,
                 ticker: str,
                 # Regime parameters
                 adx_threshold: float = 25,
                 hurst_threshold: float = 0.45,
                 keltner_threshold: float = 0.2,
                 regime_smoothing: int = 3,
                 # Position sizing parameters
                 full_size: float = 1.0,
                 half_size: float = 0.5,
                 quarter_size: float = 0.25,
                 min_confidence: float = 0.6,
                 # Signal conversion thresholds
                 long_threshold: float = 0.6,
                 short_threshold: float = 0.4):
        
        self.momentum_model = momentum_model
        self.technical_model = technical_model
        self.ticker = ticker
        
        # Regime parameters
        self.adx_threshold = adx_threshold
        self.hurst_threshold = hurst_threshold
        self.keltner_threshold = keltner_threshold
        self.regime_smoothing = regime_smoothing
        
        # Position sizing
        self.full_size = full_size
        self.half_size = half_size
        self.quarter_size = quarter_size
        self.min_confidence = min_confidence
        
        # Signal thresholds
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        
        # Storage for analysis
        self.regime_history = []
        self.signal_history = []
        
    def calculate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to calculate ensemble signals
        """
        # 1. Get individual model signals
        momentum_signals = self.momentum_model(train_data, test_data)
        technical_signals = self.technical_model(train_data, test_data)
        
        # 2. Calculate market regime
        combined_data = pd.concat([train_data, test_data])
        regime_data = classify_regime(
            combined_data,
            adx_threshold=self.adx_threshold,
            hurst_threshold=self.hurst_threshold,
            keltner_threshold=self.keltner_threshold
        )
        
        # Get regime data for test period
        test_start_idx = len(train_data)
        test_regime_score = regime_data['regime_score'].iloc[test_start_idx:]
        test_regime_label = regime_data['regime_label'].iloc[test_start_idx:]
        
        # Smooth regime scores
        test_regime_score_smooth = smooth_regime(test_regime_score, self.regime_smoothing)
        
        # 3. Convert signals to probabilities
        momentum_probs = self._signals_to_probabilities(momentum_signals)
        technical_probs = self._signals_to_probabilities(technical_signals)
        
        # 4. Calculate ensemble signals with regime weighting
        ensemble_signals = pd.DataFrame(index=test_data.index)
        ensemble_signals['signal'] = 0
        ensemble_signals['position_size'] = 0.0
        
        # Store detailed information for analysis
        details = pd.DataFrame(index=test_data.index)
        
        for i in range(len(test_data)):
            idx = test_data.index[i]
            
            # Get current values
            regime_score = test_regime_score_smooth.iloc[i]
            regime_label = test_regime_label.iloc[i]
            mom_prob = momentum_probs.iloc[i]
            tech_prob = technical_probs.iloc[i]
            mom_signal = momentum_signals.iloc[i, 0]
            tech_signal = technical_signals.iloc[i, 0]
            
            # Calculate weights
            w_mom, w_tech = get_regime_weights(regime_score)
            
            # Calculate ensemble probability
            ensemble_prob = w_mom * mom_prob + w_tech * tech_prob
            
            # Determine signal and position size
            signal, position_size = self._arbitrate_signals(
                mom_signal, tech_signal, mom_prob, tech_prob,
                regime_score, regime_label, ensemble_prob
            )
            
            ensemble_signals.iloc[i, 0] = signal
            ensemble_signals.iloc[i, 1] = position_size
            
            # Store details
            details.loc[idx, 'regime_score'] = regime_score
            details.loc[idx, 'regime_label'] = regime_label
            details.loc[idx, 'w_momentum'] = w_mom
            details.loc[idx, 'w_technical'] = w_tech
            details.loc[idx, 'momentum_prob'] = mom_prob
            details.loc[idx, 'technical_prob'] = tech_prob
            details.loc[idx, 'ensemble_prob'] = ensemble_prob
            details.loc[idx, 'momentum_signal'] = mom_signal
            details.loc[idx, 'technical_signal'] = tech_signal
            
        # Store history for analysis
        self.signal_history.append(details)
        self.regime_history.append({
            'adx': regime_data['adx'].iloc[test_start_idx:],
            'hurst': regime_data['hurst'].iloc[test_start_idx:],
            'keltner_pct': regime_data['keltner_pct'].iloc[test_start_idx:]
        })
        
        return ensemble_signals
    
    def _signals_to_probabilities(self, signals: pd.DataFrame) -> pd.Series:
        """
        Convert discrete signals (-1, 0, 1) to probabilities [0, 1]
        """
        # Simple linear mapping
        # -1 -> 0.1 (10% long probability = 90% short)
        # 0 -> 0.5 (neutral)
        # 1 -> 0.9 (90% long probability)
        probs = 0.5 + 0.4 * signals.iloc[:, 0]
        return probs
    
    def _arbitrate_signals(self, mom_signal: int, tech_signal: int,
                          mom_prob: float, tech_prob: float,
                          regime_score: float, regime_label: str,
                          ensemble_prob: float) -> Tuple[int, float]:
        """
        Arbitrate between momentum and technical signals based on regime
        
        Returns:
            (signal, position_size)
        """
        # Convert ensemble probability to signal
        if ensemble_prob >= self.long_threshold:
            ensemble_signal = 1
        elif ensemble_prob <= self.short_threshold:
            ensemble_signal = -1
        else:
            ensemble_signal = 0
        
        # Both models agree
        if mom_signal == tech_signal and mom_signal != 0:
            return mom_signal, self.full_size
        
        # Models disagree
        if mom_signal != 0 and tech_signal != 0 and mom_signal != tech_signal:
            if regime_label == 'trending':
                # Honor momentum in trends
                return mom_signal, self.half_size
            elif regime_label == 'ranging':
                # Honor technical in ranges
                return tech_signal, self.half_size
            else:
                # Uncertain regime - use ensemble but reduced size
                return ensemble_signal, self.quarter_size
        
        # Only one model has signal
        if mom_signal != 0 and tech_signal == 0:
            if regime_label == 'trending':
                return mom_signal, self.half_size
            elif regime_label == 'uncertain':
                return mom_signal, self.quarter_size
            else:
                return 0, 0.0  # Skip in ranging market
                
        if tech_signal != 0 and mom_signal == 0:
            if regime_label == 'ranging':
                return tech_signal, self.half_size
            elif regime_label == 'uncertain':
                return tech_signal, self.quarter_size
            else:
                return 0, 0.0  # Skip in trending market
        
        # Both models have no signal
        if mom_signal == 0 and tech_signal == 0:
            # Check if ensemble probability is strong enough
            if abs(ensemble_prob - 0.5) > 0.2:  # Strong ensemble signal
                return ensemble_signal, self.quarter_size
            else:
                return 0, 0.0
        
        return 0, 0.0
    
    def get_attribution_report(self) -> Dict:
        """
        Generate attribution report for the ensemble model
        """
        if not self.signal_history:
            return {}
        
        latest = self.signal_history[-1]
        
        # Calculate agreement rate
        agreement = (latest['momentum_signal'] == latest['technical_signal']).mean()
        
        # Regime distribution
        regime_dist = latest['regime_label'].value_counts(normalize=True)
        
        # Average weights by regime
        avg_weights = latest.groupby('regime_label')[['w_momentum', 'w_technical']].mean()
        
        # Signal source attribution
        signals_taken = latest[latest['ensemble_prob'] != 0.5]
        if len(signals_taken) > 0:
            momentum_driven = (signals_taken['w_momentum'] > 0.6).mean()
            technical_driven = (signals_taken['w_technical'] > 0.6).mean()
            ensemble_driven = 1 - momentum_driven - technical_driven
        else:
            momentum_driven = technical_driven = ensemble_driven = 0
        
        return {
            'agreement_rate': agreement,
            'regime_distribution': regime_dist.to_dict(),
            'average_weights_by_regime': avg_weights.to_dict(),
            'signal_attribution': {
                'momentum_driven': momentum_driven,
                'technical_driven': technical_driven,
                'ensemble_driven': ensemble_driven
            }
        }