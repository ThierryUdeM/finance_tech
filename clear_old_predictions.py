#!/usr/bin/env python3
"""
Clear old prediction data and generate fresh predictions for today
"""
import os
import json
from datetime import datetime

# Clear old prediction history
history_file = 'ChartScanAI_Shiny/evaluation_results/nvda_prediction_history.json'
if os.path.exists(history_file):
    print(f"Clearing old prediction history from {history_file}")
    with open(history_file, 'w') as f:
        json.dump([], f)
    print("✓ Old predictions cleared")

# Generate fresh predictions
print("Generating fresh predictions for today...")
from generate_nvda_predictions_robust import robust_predictions

predictions = robust_predictions()
print("✓ Fresh predictions generated")
print(f"  Current price: ${predictions['current_price']:.2f}")
print(f"  Timestamp: {predictions['timestamp']}")
print(f"  Method: {predictions['ensemble_method']}")

# Create new prediction history with today's data
new_history = [{
    'timestamp': predictions['timestamp'],
    'current_price': predictions['current_price'],
    'pred_1h_pct': predictions['pred_1h_pct'],
    'pred_1h_price': predictions['pred_1h_price'],
    'pred_1h_dir': predictions['pred_1h_dir'],
    'pred_3h_pct': predictions['pred_3h_pct'],
    'pred_3h_price': predictions['pred_3h_price'],
    'pred_3h_dir': predictions['pred_3h_dir'],
    'pred_eod_pct': predictions['pred_eod_pct'],
    'pred_eod_price': predictions['pred_eod_price'],
    'pred_eod_dir': predictions['pred_eod_dir'],
    'patterns_analyzed': predictions['patterns_analyzed'],
    'confidence': predictions['confidence'],
    'prediction_time': datetime.now().isoformat(),
    'evaluation_status': {
        '1h': 'pending',
        '3h': 'pending', 
        'eod': 'pending'
    }
}]

with open(history_file, 'w') as f:
    json.dump(new_history, f, indent=2)

print("✓ Fresh prediction history created")
print("Ready for deployment!")