#!/usr/bin/env python3
"""
Position Sizing Calculator
Calculates optimal position sizes based on Kelly Criterion and risk management rules
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List


class PositionSizingCalculator:
    """
    Calculate position sizes using multiple methods:
    1. Fixed Risk Per Trade
    2. Kelly Criterion (with safety factor)
    3. Volatility-Adjusted Sizing
    4. Confidence-Weighted Sizing
    """
    
    def __init__(self, 
                 account_value: float,
                 max_risk_per_trade: float = 0.02,
                 max_position_pct: float = 0.25,
                 kelly_fraction: float = 0.25):
        """
        Initialize calculator
        
        Args:
            account_value: Total account value
            max_risk_per_trade: Maximum risk per trade (default 2%)
            max_position_pct: Maximum position size as % of account (default 25%)
            kelly_fraction: Fraction of Kelly criterion to use (default 25%)
        """
        self.account_value = account_value
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.kelly_fraction = kelly_fraction
        
    def calculate_fixed_risk_size(self, signal: Dict) -> Dict:
        """Calculate position size based on fixed risk per trade"""
        
        if signal['signal_type'] == 'NEUTRAL':
            return self._empty_position()
        
        # Get prices
        entry_price = (signal['entry_bands']['entry_low'] + 
                      signal['entry_bands']['entry_high']) / 2
        stop_loss = signal['entry_bands']['stop_loss']
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate position size
        risk_amount = self.account_value * self.max_risk_per_trade
        shares = int(risk_amount / risk_per_share)
        
        # Apply confidence adjustment
        shares = int(shares * signal['confidence'])
        
        # Check maximum position size
        position_value = shares * entry_price
        max_position_value = self.account_value * self.max_position_pct
        
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        return {
            'method': 'fixed_risk',
            'shares': shares,
            'position_value': round(position_value, 2),
            'risk_amount': round(shares * risk_per_share, 2),
            'risk_percent': round((shares * risk_per_share) / self.account_value * 100, 2),
            'position_percent': round(position_value / self.account_value * 100, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': stop_loss,
            'risk_per_share': round(risk_per_share, 2)
        }
    
    def calculate_kelly_size(self, signal: Dict, win_rate: float = 0.5, 
                           avg_win: float = 1.5, avg_loss: float = 1.0) -> Dict:
        """
        Calculate position size using Kelly Criterion
        
        Kelly % = (p * b - q) / b
        where:
        p = probability of winning
        q = probability of losing (1 - p)
        b = ratio of win to loss
        """
        
        if signal['signal_type'] == 'NEUTRAL':
            return self._empty_position()
        
        # Adjust win rate by confidence
        adjusted_win_rate = win_rate * signal['confidence']
        
        # Calculate Kelly percentage
        p = adjusted_win_rate
        q = 1 - p
        b = avg_win / avg_loss
        
        kelly_pct = (p * b - q) / b
        
        # Apply safety factor
        kelly_pct = kelly_pct * self.kelly_fraction
        
        # Ensure positive and reasonable
        kelly_pct = max(0, min(kelly_pct, self.max_position_pct))
        
        # Calculate position
        entry_price = (signal['entry_bands']['entry_low'] + 
                      signal['entry_bands']['entry_high']) / 2
        position_value = self.account_value * kelly_pct
        shares = int(position_value / entry_price)
        
        # Calculate actual risk
        stop_loss = signal['entry_bands']['stop_loss']
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = shares * risk_per_share
        
        return {
            'method': 'kelly_criterion',
            'shares': shares,
            'position_value': round(shares * entry_price, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percent': round(risk_amount / self.account_value * 100, 2),
            'position_percent': round(kelly_pct * 100, 2),
            'kelly_percentage': round(kelly_pct * 100, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': stop_loss,
            'risk_per_share': round(risk_per_share, 2)
        }
    
    def calculate_volatility_adjusted_size(self, signal: Dict) -> Dict:
        """Calculate position size adjusted for volatility (ATR)"""
        
        if signal['signal_type'] == 'NEUTRAL':
            return self._empty_position()
        
        # Get ATR from signal
        atr = signal['entry_bands']['atr']
        entry_price = (signal['entry_bands']['entry_low'] + 
                      signal['entry_bands']['entry_high']) / 2
        
        # Calculate volatility percentage
        volatility_pct = atr / entry_price
        
        # Base position size (inverse to volatility)
        # Higher volatility = smaller position
        base_position_pct = self.max_position_pct * (1 - volatility_pct * 10)
        base_position_pct = max(0.05, min(base_position_pct, self.max_position_pct))
        
        # Adjust by confidence
        position_pct = base_position_pct * signal['confidence']
        
        # Calculate shares
        position_value = self.account_value * position_pct
        shares = int(position_value / entry_price)
        
        # Calculate risk
        stop_loss = signal['entry_bands']['stop_loss']
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = shares * risk_per_share
        
        return {
            'method': 'volatility_adjusted',
            'shares': shares,
            'position_value': round(shares * entry_price, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percent': round(risk_amount / self.account_value * 100, 2),
            'position_percent': round(position_pct * 100, 2),
            'volatility_pct': round(volatility_pct * 100, 2),
            'entry_price': round(entry_price, 2),
            'stop_loss': stop_loss,
            'risk_per_share': round(risk_per_share, 2)
        }
    
    def calculate_all_methods(self, signal: Dict) -> Dict:
        """Calculate position sizes using all methods"""
        
        return {
            'ticker': signal['ticker'],
            'signal_type': signal['signal_type'],
            'confidence': signal['confidence'],
            'methods': {
                'fixed_risk': self.calculate_fixed_risk_size(signal),
                'kelly': self.calculate_kelly_size(signal),
                'volatility_adjusted': self.calculate_volatility_adjusted_size(signal)
            },
            'recommended': self._get_recommended_size(signal)
        }
    
    def _get_recommended_size(self, signal: Dict) -> Dict:
        """Get recommended position size based on signal characteristics"""
        
        # Calculate all methods
        fixed_risk = self.calculate_fixed_risk_size(signal)
        kelly = self.calculate_kelly_size(signal)
        vol_adjusted = self.calculate_volatility_adjusted_size(signal)
        
        # For high confidence signals, use max of fixed risk and Kelly
        if signal['confidence'] >= 0.7:
            if kelly['shares'] > fixed_risk['shares']:
                recommended = kelly.copy()
                recommended['reason'] = 'High confidence - Kelly optimal'
            else:
                recommended = fixed_risk.copy()
                recommended['reason'] = 'High confidence - Fixed risk optimal'
        
        # For medium confidence, use minimum of all methods
        elif signal['confidence'] >= 0.5:
            sizes = [fixed_risk['shares'], kelly['shares'], vol_adjusted['shares']]
            min_shares = min(sizes)
            
            if min_shares == fixed_risk['shares']:
                recommended = fixed_risk.copy()
            elif min_shares == kelly['shares']:
                recommended = kelly.copy()
            else:
                recommended = vol_adjusted.copy()
            
            recommended['reason'] = 'Medium confidence - Conservative sizing'
        
        # For low confidence, use volatility adjusted
        else:
            recommended = vol_adjusted.copy()
            recommended['reason'] = 'Low confidence - Volatility adjusted'
        
        return recommended
    
    def _empty_position(self) -> Dict:
        """Return empty position structure"""
        return {
            'shares': 0,
            'position_value': 0,
            'risk_amount': 0,
            'risk_percent': 0,
            'position_percent': 0,
            'entry_price': 0,
            'stop_loss': 0,
            'risk_per_share': 0
        }


def create_position_report(signals_file: str, account_value: float) -> Dict:
    """Create complete position sizing report for all signals"""
    
    # Load signals
    with open(signals_file, 'r') as f:
        signals = json.load(f)
    
    # Initialize calculator
    calc = PositionSizingCalculator(account_value)
    
    # Calculate positions for all signals
    positions = []
    total_allocation = 0
    total_risk = 0
    
    for signal in signals:
        if signal['signal_type'] != 'NEUTRAL':
            position = calc.calculate_all_methods(signal)
            positions.append(position)
            
            # Track totals
            rec = position['recommended']
            total_allocation += rec['position_value']
            total_risk += rec['risk_amount']
    
    # Create report
    report = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'account_value': account_value,
        'total_signals': len([s for s in signals if s['signal_type'] != 'NEUTRAL']),
        'total_allocation': round(total_allocation, 2),
        'total_allocation_pct': round(total_allocation / account_value * 100, 2),
        'total_risk': round(total_risk, 2),
        'total_risk_pct': round(total_risk / account_value * 100, 2),
        'positions': positions,
        'summary_by_ticker': _create_ticker_summary(positions)
    }
    
    return report


def _create_ticker_summary(positions: List[Dict]) -> List[Dict]:
    """Create summary by ticker"""
    
    summary = []
    for pos in positions:
        rec = pos['recommended']
        summary.append({
            'ticker': pos['ticker'],
            'action': pos['signal_type'],
            'shares': rec['shares'],
            'entry': rec['entry_price'],
            'stop': rec['stop_loss'],
            'position_value': rec['position_value'],
            'risk': rec['risk_amount'],
            'method': rec.get('method', 'recommended'),
            'reason': rec.get('reason', '')
        })
    
    return summary


def main():
    """Example usage"""
    
    # Example parameters
    account_value = 100000  # $100k account
    
    # Create position report
    report = create_position_report('trading_signals_latest.json', account_value)
    
    # Save report
    with open('position_sizing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("Position Sizing Report")
    print("=" * 60)
    print(f"Account Value: ${account_value:,.2f}")
    print(f"Total Signals: {report['total_signals']}")
    print(f"Total Allocation: ${report['total_allocation']:,.2f} ({report['total_allocation_pct']:.1f}%)")
    print(f"Total Risk: ${report['total_risk']:,.2f} ({report['total_risk_pct']:.1f}%)")
    print("\nPositions by Ticker:")
    print("-" * 60)
    
    for pos in report['summary_by_ticker']:
        print(f"{pos['ticker']:<6} {pos['action']:<5} {pos['shares']:>6} shares @ ${pos['entry']:>7.2f}")
        print(f"       Stop: ${pos['stop']:>7.2f} | Risk: ${pos['risk']:>8.2f} | Method: {pos['method']}")
        print(f"       {pos['reason']}")
        print()


if __name__ == "__main__":
    main()