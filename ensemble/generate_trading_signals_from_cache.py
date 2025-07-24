#!/usr/bin/env python3
"""
Generate trading signals from cached ensemble model outputs
This avoids re-fetching data from Yahoo Finance
"""

import json
import os
from datetime import datetime


def load_ensemble_outputs():
    """Load the ensemble model outputs from the previous step"""
    try:
        with open('../ensemble_model_outputs.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ensemble outputs: {str(e)}")
        return None


def create_signal_card_from_output(model_output):
    """Create a signal card from model output"""
    ticker = model_output['ticker']
    signal = model_output['signal']
    position_size = model_output['position_size']
    latest_price = model_output['latest_price']
    vwap = model_output['vwap']
    regime = model_output['regime']
    
    # Determine signal type
    if signal > 0:
        signal_type = "BUY"
    elif signal < 0:
        signal_type = "SELL"
    else:
        signal_type = "NEUTRAL"
    
    # Create entry bands (simplified version)
    entry_margin = latest_price * 0.001  # 0.1% margin
    stop_margin = latest_price * 0.02    # 2% stop loss
    target_margin = latest_price * 0.03  # 3% target
    
    if signal > 0:  # Buy signal
        entry_bands = {
            'entry_low': round(latest_price - entry_margin, 2),
            'entry_high': round(latest_price + entry_margin, 2),
            'stop_loss': round(latest_price - stop_margin, 2),
            'target_1h': round(latest_price + target_margin * 0.33, 2),
            'target_3h': round(latest_price + target_margin * 0.66, 2),
            'target_eod': round(latest_price + target_margin, 2),
            'atr': round(stop_margin, 2)
        }
    elif signal < 0:  # Sell signal
        entry_bands = {
            'entry_low': round(latest_price - entry_margin, 2),
            'entry_high': round(latest_price + entry_margin, 2),
            'stop_loss': round(latest_price + stop_margin, 2),
            'target_1h': round(latest_price - target_margin * 0.33, 2),
            'target_3h': round(latest_price - target_margin * 0.66, 2),
            'target_eod': round(latest_price - target_margin, 2),
            'atr': round(stop_margin, 2)
        }
    else:
        entry_bands = None
    
    # Create signal card
    signal_card = {
        'ticker': ticker,
        'timestamp': model_output['timestamp'],
        'signal_type': signal_type,
        'confidence': position_size,
        'latest_price': latest_price,
        'vwap': vwap,
        'volume_ratio': model_output['volume_ratio'],
        'market_regime': regime,
        'entry_bands': entry_bands,
        'model_agreement': "Strong Agreement" if position_size >= 0.8 else "Moderate Agreement" if position_size >= 0.5 else "Weak Agreement",
        'risk_reward': {
            '1h': 0.33 if entry_bands else 0,
            '3h': 0.66 if entry_bands else 0,
            'eod': 1.0 if entry_bands else 0
        }
    }
    
    return signal_card


def create_actionable_summary(signal_cards):
    """Create actionable summary from signal cards"""
    # Filter active signals (non-neutral)
    active_signals = [s for s in signal_cards if s['signal_type'] != 'NEUTRAL']
    
    # Sort by confidence
    active_signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_signals': len(signal_cards),
        'active_signals': len(active_signals),
        'buy_signals': len([s for s in active_signals if s['signal_type'] == 'BUY']),
        'sell_signals': len([s for s in active_signals if s['signal_type'] == 'SELL']),
        'high_confidence': len([s for s in active_signals if s['confidence'] >= 0.5]),
        'top_opportunities': []
    }
    
    # Add top 3 opportunities
    for signal in active_signals[:3]:
        if signal['entry_bands']:
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


def main():
    """Main function to generate trading signals from cached outputs"""
    
    print(f"Generating trading signals at {datetime.now()}")
    print("=" * 60)
    
    # Load ensemble outputs
    ensemble_data = load_ensemble_outputs()
    if not ensemble_data:
        print("ERROR: Could not load ensemble outputs")
        return
    
    signal_cards = []
    
    # Process each model output
    for model_output in ensemble_data.get('model_outputs', []):
        ticker = model_output['ticker']
        print(f"\nProcessing {ticker}...")
        
        # Generate signal card
        card = create_signal_card_from_output(model_output)
        signal_cards.append(card)
        
        print(f"  Signal: {card['signal_type']} (Confidence: {card['confidence']:.2f})")
        if card['signal_type'] != 'NEUTRAL' and card['entry_bands']:
            print(f"  Entry: ${card['entry_bands']['entry_low']}-${card['entry_bands']['entry_high']}")
            print(f"  Stop: ${card['entry_bands']['stop_loss']}")
    
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
        print("\nTOP OPPORTUNITIES:")
        for i, opp in enumerate(summary['top_opportunities'], 1):
            print(f"\n{i}. {opp['ticker']} - {opp['action']}")
            print(f"   Confidence: {opp['confidence']:.2f}")
            print(f"   Entry: {opp['entry_range']}")
            print(f"   Stop: {opp['stop']}")
            print(f"   Target (1h): {opp['target_1h']}")


if __name__ == "__main__":
    main()