#!/usr/bin/env python3
import json
import os

if os.path.exists('evaluation_results/nvda_prediction_history.json'):
    with open('evaluation_results/nvda_prediction_history.json', 'r') as f:
        history = json.load(f)
        if history:
            latest = history[-1]
            print(f'Time: {latest["timestamp"]}')
            print(f'Price: ${latest["current_price"]:.2f}')
            print(f'1H: {latest["pred_1h_pct"]:+.3f}% ({latest["pred_1h_dir"]})')
            print(f'3H: {latest["pred_3h_pct"]:+.3f}% ({latest["pred_3h_dir"]})')
            print(f'EOD: {latest["pred_eod_pct"]:+.3f}% ({latest["pred_eod_dir"]})')