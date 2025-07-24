#!/usr/bin/env python3
"""
Generate Trading Signals from Ensemble Models
Produces actionable entry/exit signals with precise price levels
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ensemble models
from walk_forward_tests.model.hybrid_momentum_technical.ensemble_nvda import ensemble_nvda
from walk_forward_tests.model.hybrid_momentum_technical.ensemble_tsla import ensemble_tsla
from walk_forward_tests.model.hybrid_momentum_technical.ensemble_aapl_v2 import ensemble_aapl_v2
from walk_forward_tests.model.hybrid_momentum_technical.ensemble_msft_v2 import ensemble_msft_v2

# Import market regime classification
from walk_forward_tests.model.hybrid_momentum_technical.market_regime import classify_regime


def fetch_latest_data(ticker, lookback_days=90):
    """Fetch latest 15-minute data for ticker"""
    try:
        # Calculate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Download data
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='15m',
            progress=False
        )
        
        if data.empty:
            print(f"Warning: No data retrieved for {ticker}")
            return None
            
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        data.reset_index(inplace=True)
        data.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        
        # Add required calculations
        data['returns'] = data['close'].pct_change()
        data['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None


def calculate_entry_bands(data, signal, regime_data):
    """Calculate precise entry price bands based on signal and market conditions"""
    
    current_price = data['close'].iloc[-1]
    vwap = data['vwap'].iloc[-1]
    
    # Calculate ATR for volatility bands
    high = data['high']
    low = data['low']
    close = data['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean().iloc[-1]
    
    # Calculate entry bands based on signal direction
    if signal > 0:  # Buy signal
        # Entry range: Between current price and VWAP, adjusted by ATR
        entry_low = min(current_price, vwap) - 0.1 * atr
        entry_high = max(current_price, vwap) + 0.1 * atr
        
        # Stop loss: Below entry range
        stop_loss = entry_low - 0.5 * atr
        
        # Targets based on ATR multiples
        target_1h = entry_high + 0.5 * atr
        target_3h = entry_high + 1.0 * atr
        target_eod = entry_high + 1.5 * atr
        
    elif signal < 0:  # Sell signal
        # Entry range for shorts
        entry_low = min(current_price, vwap) - 0.1 * atr
        entry_high = max(current_price, vwap) + 0.1 * atr
        
        # Stop loss: Above entry range
        stop_loss = entry_high + 0.5 * atr
        
        # Targets (downside)
        target_1h = entry_low - 0.5 * atr
        target_3h = entry_low - 1.0 * atr
        target_eod = entry_low - 1.5 * atr
        
    else:  # No signal
        return None
    
    return {
        'entry_low': round(entry_low, 2),
        'entry_high': round(entry_high, 2),
        'stop_loss': round(stop_loss, 2),
        'target_1h': round(target_1h, 2),
        'target_3h': round(target_3h, 2),
        'target_eod': round(target_eod, 2),
        'atr': round(atr, 2)
    }


def generate_signal_card(ticker, model_func, data):
    """Generate comprehensive signal card for a ticker"""
    
    if data is None or len(data) < 200:
        return None
    
    # Split data for walk-forward simulation
    train_size = 60 * 26  # 60 days of 15min bars
    if len(data) < train_size + 26:
        return None
        
    train_data = data.iloc[-train_size-26:-26].copy()
    test_data = data.iloc[-26:].copy()  # Last day
    
    # Get ensemble signal
    try:
        signals = model_func(train_data, test_data)
        
        # Extract signal and confidence
        if 'signal' in signals.columns:
            signal = signals['signal'].iloc[-1]
        else:
            signal = signals.iloc[-1, 0]
            
        # Get position size if available
        if 'position_size' in signals.columns:
            position_size = signals['position_size'].iloc[-1]
        else:
            position_size = 1.0 if signal != 0 else 0.0
            
    except Exception as e:
        print(f"Error generating signal for {ticker}: {str(e)}")
        return None
    
    # Calculate market regime
    regime_data = classify_regime(test_data)
    regime_label = regime_data['regime_label'].iloc[-1]
    regime_score = regime_data['regime_score'].iloc[-1]
    
    # Calculate entry bands
    entry_info = calculate_entry_bands(test_data, signal, regime_data)
    
    if entry_info is None:
        signal_type = "NEUTRAL"
        model_agreement = "None"
    else:
        signal_type = "BUY" if signal > 0 else "SELL"
        # Determine model agreement based on position size
        if position_size >= 1.0:
            model_agreement = "Strong Agreement"
        elif position_size >= 0.5:
            model_agreement = "Moderate Agreement"
        elif position_size > 0:
            model_agreement = "Weak Agreement"
        else:
            model_agreement = "Conflict"
    
    # Create signal card
    signal_card = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'signal_type': signal_type,
        'signal_strength': float(abs(signal)),
        'confidence': float(position_size),
        'current_price': round(test_data['close'].iloc[-1], 2),
        'vwap': round(test_data['vwap'].iloc[-1], 2),
        'volume_ratio': round(test_data['volume'].iloc[-1] / test_data['volume'].iloc[-26:-1].mean(), 2),
        'market_regime': {
            'label': regime_label,
            'score': round(float(regime_score), 2),
            'adx': round(float(regime_data['adx'].iloc[-1]), 2),
            'hurst': round(float(regime_data['hurst'].iloc[-1]), 3),
            'keltner_position': round(float(regime_data['keltner_pct'].iloc[-1]), 2)
        },
        'model_agreement': model_agreement,
        'validity_bars': 3,  # Valid for 45 minutes
        'expires_at': (datetime.now() + timedelta(minutes=45)).isoformat()
    }
    
    # Add entry information if signal exists
    if entry_info:
        signal_card['entry_bands'] = entry_info
        signal_card['risk_reward'] = {
            '1h': round(abs(entry_info['target_1h'] - entry_info['entry_high']) / 
                       abs(entry_info['stop_loss'] - entry_info['entry_high']), 2),
            '3h': round(abs(entry_info['target_3h'] - entry_info['entry_high']) / 
                       abs(entry_info['stop_loss'] - entry_info['entry_high']), 2),
            'eod': round(abs(entry_info['target_eod'] - entry_info['entry_high']) / 
                        abs(entry_info['stop_loss'] - entry_info['entry_high']), 2)
        }
    
    return signal_card


def create_actionable_summary(signal_cards):
    """Create summary of top actionable opportunities"""
    
    # Filter for active signals only
    active_signals = [
        card for card in signal_cards 
        if card and card['signal_type'] != 'NEUTRAL'
    ]
    
    # Sort by confidence
    active_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Create summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'market_status': 'OPEN' if is_market_open() else 'CLOSED',
        'total_signals': len(active_signals),
        'buy_signals': len([s for s in active_signals if s['signal_type'] == 'BUY']),
        'sell_signals': len([s for s in active_signals if s['signal_type'] == 'SELL']),
        'high_confidence': len([s for s in active_signals if s['confidence'] >= 0.5]),
        'top_opportunities': []
    }
    
    # Add top 3 opportunities
    for signal in active_signals[:3]:
        opportunity = {
            'ticker': signal['ticker'],
            'action': signal['signal_type'],
            'confidence': signal['confidence'],
            'entry_range': f"${signal['entry_bands']['entry_low']}-${signal['entry_bands']['entry_high']}",
            'stop': f"${signal['entry_bands']['stop_loss']}",
            'target_1h': f"${signal['entry_bands']['target_1h']}",
            'risk_reward_1h': signal['risk_reward']['1h'],
            'regime': signal['market_regime']['label']
        }
        summary['top_opportunities'].append(opportunity)
    
    return summary


def is_market_open():
    """Check if US market is currently open"""
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:
        return False
    
    # Market hours: 9:30 AM - 4:00 PM ET
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # Adjust for timezone if needed (assuming system is in ET)
    return market_open <= now <= market_close


def main():
    """Main function to generate all trading signals"""
    
    # Model configurations
    models = {
        'NVDA': ensemble_nvda,
        'TSLA': ensemble_tsla,
        'AAPL': ensemble_aapl_v2,
        'MSFT': ensemble_msft_v2
    }
    
    print(f"Generating trading signals at {datetime.now()}")
    print("=" * 60)
    
    signal_cards = []
    
    # Generate signals for each ticker
    for ticker, model_func in models.items():
        print(f"\nProcessing {ticker}...")
        
        # Fetch latest data
        data = fetch_latest_data(ticker)
        if data is None:
            continue
        
        # Generate signal card
        card = generate_signal_card(ticker, model_func, data)
        if card:
            signal_cards.append(card)
            print(f"  Signal: {card['signal_type']} (Confidence: {card['confidence']:.2f})")
            if card['signal_type'] != 'NEUTRAL':
                print(f"  Entry: ${card['entry_bands']['entry_low']}-${card['entry_bands']['entry_high']}")
                print(f"  Stop: ${card['entry_bands']['stop_loss']}")
        else:
            print(f"  No signal generated")
    
    # Save detailed signals
    output_file = 'trading_signals_latest.json'
    with open(output_file, 'w') as f:
        json.dump(signal_cards, f, indent=2)
    print(f"\nDetailed signals saved to {output_file}")
    
    # Create and save summary
    summary = create_actionable_summary(signal_cards)
    summary_file = 'actionable_summary_latest.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    print(f"Total Signals: {summary['total_signals']}")
    print(f"Buy Signals: {summary['buy_signals']}")
    print(f"Sell Signals: {summary['sell_signals']}")
    print(f"High Confidence: {summary['high_confidence']}")
    
    if summary['top_opportunities']:
        print("\nTop Opportunities:")
        for i, opp in enumerate(summary['top_opportunities'], 1):
            print(f"\n{i}. {opp['ticker']} - {opp['action']}")
            print(f"   Confidence: {opp['confidence']:.1%}")
            print(f"   Entry: {opp['entry_range']}")
            print(f"   Stop: {opp['stop']}")
            print(f"   Target (1h): {opp['target_1h']} (RR: {opp['risk_reward_1h']})")


if __name__ == "__main__":
    main()