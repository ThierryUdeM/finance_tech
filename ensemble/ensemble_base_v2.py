#!/usr/bin/env python3
"""
Ensemble Model Base Class V2
Improved version with logistic probability mapping, ticker-specific regime weights,
and enhanced risk management
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

from market_regime import classify_regime, smooth_regime


class EnsembleModelV2:
    """
    Improved ensemble model with better probability calibration and risk management
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
                 quarter_size: float = 0.15,  # Reduced from 0.25
                 min_confidence: float = 0.65,  # Increased from 0.6
                 # Signal conversion thresholds
                 long_threshold: float = 0.65,  # Increased for MSFT/AAPL
                 short_threshold: float = 0.35,  # Decreased for MSFT/AAPL
                 # Risk management
                 max_daily_trades: int = 2,
                 atr_stop_multiplier: float = 0.5,
                 trailing_start_r: float = 0.5,
                 portfolio_stop_r: float = 2.0,
                 # Ticker-specific regime weight function
                 regime_weight_func: Optional[Callable] = None,
                 # Probability calibration strength
                 prob_calibration_strength: float = 1.5):
        
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
        
        # Risk management
        self.max_daily_trades = max_daily_trades
        self.atr_stop_multiplier = atr_stop_multiplier
        self.trailing_start_r = trailing_start_r
        self.portfolio_stop_r = portfolio_stop_r
        
        # Regime weight function (ticker-specific)
        self.regime_weight_func = regime_weight_func or self._default_regime_weights
        
        # Probability calibration
        self.prob_calibration_strength = prob_calibration_strength
        
        # Storage for analysis
        self.regime_history = []
        self.signal_history = []
        self.daily_trade_count = {}
        self.daily_pnl = {}
        
    def calculate_signals(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to calculate ensemble signals with improved risk management
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
        
        # 3. Convert signals to probabilities with improved calibration
        momentum_probs = self._signals_to_probabilities_logistic(momentum_signals)
        technical_probs = self._signals_to_probabilities_logistic(technical_signals)
        
        # 4. Calculate ATR for stop-loss sizing
        atr = self._calculate_atr(test_data, period=14)
        
        # 5. Calculate ensemble signals with regime weighting
        ensemble_signals = pd.DataFrame(index=test_data.index)
        ensemble_signals['signal'] = 0
        ensemble_signals['position_size'] = 0.0
        ensemble_signals['stop_price'] = 0.0
        ensemble_signals['trailing_stop'] = 0.0
        
        # Store detailed information for analysis
        details = pd.DataFrame(index=test_data.index)
        
        for i in range(len(test_data)):
            idx = test_data.index[i]
            current_date = idx.date()
            
            # Check daily trade limit
            if current_date not in self.daily_trade_count:
                self.daily_trade_count[current_date] = 0
                self.daily_pnl[current_date] = 0.0
            
            # Skip if we've hit daily trade limit
            if self.daily_trade_count[current_date] >= self.max_daily_trades:
                ensemble_signals.iloc[i, 0] = 0
                ensemble_signals.iloc[i, 1] = 0.0
                continue
            
            # Skip if portfolio stop hit for the day
            if self.daily_pnl[current_date] <= -self.portfolio_stop_r:
                ensemble_signals.iloc[i, 0] = 0
                ensemble_signals.iloc[i, 1] = 0.0
                continue
            
            # Get current values
            regime_score = test_regime_score_smooth.iloc[i]
            regime_label = test_regime_label.iloc[i]
            mom_prob = momentum_probs.iloc[i]
            tech_prob = technical_probs.iloc[i]
            mom_signal = momentum_signals.iloc[i, 0]
            tech_signal = technical_signals.iloc[i, 0]
            
            # Calculate weights using ticker-specific function
            w_mom, w_tech = self.regime_weight_func(regime_score)
            
            # Calculate ensemble probability
            ensemble_prob = w_mom * mom_prob + w_tech * tech_prob
            
            # Determine signal and position size
            signal, position_size = self._arbitrate_signals_improved(
                mom_signal, tech_signal, mom_prob, tech_prob,
                regime_score, regime_label, ensemble_prob
            )
            
            # Set stop prices if we have a signal
            if signal != 0 and position_size > 0:
                entry_price = test_data['close'].iloc[i]
                atr_value = atr.iloc[i] if not pd.isna(atr.iloc[i]) else test_data['close'].iloc[i] * 0.01
                
                # Initial stop loss
                stop_distance = self.atr_stop_multiplier * atr_value
                ensemble_signals.iloc[i, 2] = entry_price - signal * stop_distance
                
                # Trailing stop activation level
                ensemble_signals.iloc[i, 3] = entry_price + signal * self.trailing_start_r * atr_value
                
                # Increment trade count
                self.daily_trade_count[current_date] += 1
            
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
    
    def _signals_to_probabilities_logistic(self, signals: pd.DataFrame) -> pd.Series:
        """
        Convert discrete signals (-1, 0, 1) to probabilities using logistic function
        This provides better separation than linear mapping
        """
        # Use logistic (sigmoid) function for smoother probability mapping
        # expit(1.5 * signal) gives approximately:
        # -1 -> 0.18 (18% long probability)
        # 0 -> 0.5 (neutral)
        # 1 -> 0.82 (82% long probability)
        probs = expit(self.prob_calibration_strength * signals.iloc[:, 0])
        return probs
    
    def _default_regime_weights(self, regime_score: float) -> Tuple[float, float]:
        """
        Default regime weight function (can be overridden per ticker)
        """
        # Linear interpolation between momentum and technical
        # regime_score: -1 (ranging) to +1 (trending)
        w_momentum = 0.5 + 0.4 * max(0, regime_score)  # 0.1 to 0.9
        w_technical = 1.0 - w_momentum
        return w_momentum, w_technical
    
    def _arbitrate_signals_improved(self, mom_signal: int, tech_signal: int,
                                   mom_prob: float, tech_prob: float,
                                   regime_score: float, regime_label: str,
                                   ensemble_prob: float) -> Tuple[int, float]:
        """
        Improved signal arbitration with confidence-based position sizing
        """
        # Calculate confidence as distance from neutral (0.5)
        confidence = abs(ensemble_prob - 0.5) * 2  # Scale to [0, 1]
        
        # Convert ensemble probability to signal
        if ensemble_prob >= self.long_threshold:
            ensemble_signal = 1
        elif ensemble_prob <= self.short_threshold:
            ensemble_signal = -1
        else:
            ensemble_signal = 0
        
        # Both models strongly agree
        if mom_signal == tech_signal and mom_signal != 0:
            # Check if both have high confidence
            if abs(mom_prob - 0.5) > 0.25 and abs(tech_prob - 0.5) > 0.25:
                return mom_signal, self.full_size
            else:
                return mom_signal, self.half_size
        
        # Models disagree
        if mom_signal != 0 and tech_signal != 0 and mom_signal != tech_signal:
            # Only trade if ensemble confidence is high
            if confidence >= self.min_confidence:
                if regime_label == 'trending':
                    return mom_signal, self.quarter_size
                elif regime_label == 'ranging':
                    return tech_signal, self.quarter_size
                else:
                    return 0, 0.0  # Skip uncertain regimes with disagreement
            else:
                return 0, 0.0
        
        # Only one model has signal
        if mom_signal != 0 and tech_signal == 0:
            if regime_label == 'trending' and abs(mom_prob - 0.5) > 0.3:
                return mom_signal, self.half_size
            elif regime_label == 'uncertain' and confidence >= self.min_confidence:
                return mom_signal, self.quarter_size
            else:
                return 0, 0.0
                
        if tech_signal != 0 and mom_signal == 0:
            if regime_label == 'ranging' and abs(tech_prob - 0.5) > 0.3:
                return tech_signal, self.half_size
            elif regime_label == 'uncertain' and confidence >= self.min_confidence:
                return tech_signal, self.quarter_size
            else:
                return 0, 0.0
        
        # Both models have no signal
        if mom_signal == 0 and tech_signal == 0:
            # Only trade if ensemble has very high confidence
            if confidence >= self.min_confidence * 1.2:
                return ensemble_signal, self.quarter_size
            else:
                return 0, 0.0
        
        return 0, 0.0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range for stop-loss sizing
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        return atr
    
    def get_attribution_report(self) -> Dict:
        """
        Generate enhanced attribution report
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
        
        # Confidence distribution
        confidence_scores = abs(latest['ensemble_prob'] - 0.5) * 2
        avg_confidence = confidence_scores.mean()
        high_confidence_rate = (confidence_scores > self.min_confidence).mean()
        
        return {
            'agreement_rate': agreement,
            'regime_distribution': regime_dist.to_dict(),
            'average_weights_by_regime': avg_weights.to_dict(),
            'signal_attribution': {
                'momentum_driven': momentum_driven,
                'technical_driven': technical_driven,
                'ensemble_driven': ensemble_driven
            },
            'confidence_metrics': {
                'average_confidence': avg_confidence,
                'high_confidence_rate': high_confidence_rate,
                'trades_per_day': sum(self.daily_trade_count.values()) / max(1, len(self.daily_trade_count))
            }
        }