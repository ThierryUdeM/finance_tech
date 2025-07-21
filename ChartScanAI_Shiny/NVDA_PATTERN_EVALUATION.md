# NVDA Pattern Prediction Evaluation

This module evaluates the performance of NVDA price predictions using volatility-based pattern analysis.

## Overview

The evaluation system tests the accuracy of 1-hour, 3-hour, and end-of-day (EOD) price predictions for NVIDIA (NVDA) stock using historical 15-minute bar data.

## Components

### 1. Prediction Script (`generate_nvda_predictions_simple.py`)
- Loads real NVDA price data from CSV
- Calculates recent volatility from the last 20 bars (5 hours)
- Generates predictions for 1h, 3h, and EOD timeframes
- Uses statistical modeling based on historical volatility patterns

### 2. Evaluation Script (`evaluate_nvda_patterns.py`)
- Runs backtests on historical data
- Compares predictions with actual price movements
- Calculates accuracy metrics for:
  - Direction accuracy (BULLISH/BEARISH/NEUTRAL)
  - Average prediction error
  - Accuracy by prediction type

### 3. Performance Dashboard (`performance_dashboard.py`)
- Generates visual performance reports
- Creates trend charts showing accuracy over time
- Produces HTML dashboard with comprehensive metrics

### 4. GitHub Actions Workflow (`.github/workflows/evaluate_nvda_patterns.yml`)
- Runs daily evaluations after market close
- Tracks performance over time
- Creates issues if performance drops below threshold

## Metrics

The evaluation tracks the following key metrics:

### Direction Accuracy
- Measures how often the predicted direction (up/down/neutral) matches actual movement
- Threshold: >0.1% for BULLISH/BEARISH classification

### Average Prediction Error
- Average absolute difference between predicted and actual percentage changes

### Accuracy by Direction
- Separate accuracy tracking for BULLISH, BEARISH, and NEUTRAL predictions
- Helps identify if the model has directional bias

## Usage

### Running Locally

```bash
# Run evaluation
cd ChartScanAI_Shiny
python evaluate_nvda_patterns.py

# Generate performance dashboard
python performance_dashboard.py
```

### Expected Performance

Based on financial prediction standards:
- **Good**: >55% direction accuracy
- **Acceptable**: 45-55% direction accuracy  
- **Poor**: <45% direction accuracy

Note: Financial markets are inherently unpredictable. Even 55% accuracy can be profitable with proper risk management.

## Output Files

### Evaluation Results
- `evaluation_results/nvda_eval_1h_[timestamp].csv` - Detailed 1-hour predictions
- `evaluation_results/nvda_eval_3h_[timestamp].csv` - Detailed 3-hour predictions
- `evaluation_results/nvda_eval_eod_[timestamp].csv` - Detailed EOD predictions
- `evaluation_results/nvda_metrics_[timestamp].json` - Summary metrics
- `evaluation_results/nvda_performance_report_[timestamp].md` - Markdown report

### Performance Dashboard
- `performance_dashboard.html` - Interactive HTML dashboard
- `performance_charts/performance_trends.png` - Trend visualization

## Configuration

### Backtest Parameters
- `num_days`: Number of days to backtest (default: 30)
- `predictions_per_day`: Number of predictions per day (default: 10)

### Performance Thresholds
- Overall accuracy threshold: 45% (configurable in workflow)
- Individual timeframe thresholds can be adjusted

## Integration with Shiny App

The predictions are saved to `nvda_predictions.csv` which is read by the Shiny dashboard (`multi_ticker_monitor_azure.R`) to display:
- Current predictions for 1h, 3h, and EOD
- Price targets
- Confidence levels
- Pattern analysis details

## Data Requirements

- NVDA 15-minute bar data in CSV format
- Columns required: timestamp (index), Open, High, Low, Close, Volume
- Minimum history: 30 days for meaningful backtesting

## Future Improvements

1. **Enhanced Pattern Matching**: Integrate full sklearn-based pattern matching when dependencies are available
2. **Multiple Tickers**: Extend to other stocks beyond NVDA
3. **Feature Engineering**: Add technical indicators and market regime detection
4. **Real-time Updates**: Stream predictions during market hours
5. **Risk Metrics**: Add Sharpe ratio and maximum drawdown calculations