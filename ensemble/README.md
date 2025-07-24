# Trading Signal Generation System

## Overview

This system generates actionable trading signals using ensemble models that combine momentum and technical analysis strategies. It produces precise entry/exit levels, position sizing recommendations, and risk management parameters.

## System Components

### 1. Signal Generator (`generate_trading_signals.py`)

The main script that:
- Fetches latest 15-minute bar data for NVDA, TSLA, AAPL, and MSFT
- Runs ensemble models to generate trading signals
- Calculates precise entry bands based on VWAP and ATR
- Determines stop-loss levels and profit targets
- Outputs JSON files with detailed signal cards

### 2. Position Sizing Calculator (`position_sizing_calculator.py`)

Calculates optimal position sizes using three methods:
- **Fixed Risk**: 2% risk per trade
- **Kelly Criterion**: Optimal sizing based on win rate
- **Volatility Adjusted**: Smaller positions for higher volatility

### 3. R Shiny Display Functions (`trading_signal_functions.R`)

Helper functions for displaying signals in R Shiny:
- Signal cards with color-coded buy/sell zones
- Active signals table
- Position sizing recommendations
- Market regime overview
- High-confidence alerts

### 4. GitHub Actions Workflow (`.github/workflows/generate_trading_signals.yml`)

Automated signal generation:
- Runs every 15 minutes during market hours
- Checks if market is open before executing
- Uploads results to Azure blob storage
- Can be manually triggered for testing

## Signal Structure

Each signal contains:

```json
{
  "ticker": "NVDA",
  "signal_type": "BUY",
  "confidence": 0.75,
  "current_price": 131.25,
  "entry_bands": {
    "entry_low": 131.04,
    "entry_high": 131.45,
    "stop_loss": 130.63,
    "target_1h": 132.07,
    "target_3h": 133.36,
    "target_eod": 134.65,
    "atr": 0.82
  },
  "market_regime": {
    "label": "trending",
    "score": 0.65
  },
  "model_agreement": "Strong Agreement",
  "validity_bars": 3,
  "expires_at": "2024-01-15T10:45:00"
}
```

## Model Configuration

The system uses optimized model combinations:

| Ticker | Model Type | Strategy |
|--------|------------|----------|
| NVDA   | V1 Original | High-frequency momentum |
| TSLA   | V1 Original | Aggressive momentum capture |
| AAPL   | V2 Improved | Conservative mean-reversion |
| MSFT   | V2 Improved | Selective quality filters |

## Usage

### Manual Execution

```bash
# Generate signals manually
cd github_ready
python generate_trading_signals.py

# Calculate position sizes
python position_sizing_calculator.py
```

### Automated Execution

The GitHub Actions workflow runs automatically every 15 minutes during market hours. To trigger manually:

1. Go to Actions tab in GitHub
2. Select "Generate Trading Signals" workflow
3. Click "Run workflow"

### R Shiny Integration

```r
# Load the helper functions
source("trading_signal_functions.R")

# Load latest signals
signals <- load_trading_signals()
summary <- load_signal_summary()

# Display signal cards
lapply(signals, create_signal_card)

# Create signals table
signals_table <- create_signals_table(signals)

# Calculate position size
position <- calculate_position_size(
  account_value = 100000,
  signal = signals[[1]],
  risk_per_trade = 0.02
)
```

## Output Files

1. **trading_signals_latest.json**: Complete signal details for all tickers
2. **actionable_summary_latest.json**: Summary of top opportunities
3. **position_sizing_report.json**: Position sizing recommendations

## Risk Management

### Position Sizing Rules
- Maximum 2% risk per trade
- Maximum 25% allocation per position
- Confidence-based scaling (0.25x to 1.0x)
- Volatility adjustments for high ATR

### Signal Validity
- Signals expire after 3 bars (45 minutes)
- Entry bands provide precise zones, not single prices
- Stop-loss levels are volatility-adjusted

### Market Regime Adaptation
- **Trending**: Favor momentum signals
- **Ranging**: Favor technical/mean-reversion
- **Uncertain**: Reduce position sizes

## Setup Requirements

### Python Dependencies
```bash
pip install pandas numpy yfinance scipy scikit-learn TA-Lib
```

### R Dependencies
```r
install.packages(c("jsonlite", "DT", "shiny", "dplyr"))
```

### Azure Storage
Set the `AZURE_STORAGE_CONNECTION_STRING` secret in GitHub repository settings.

## Performance Expectations

Based on backtesting:
- NVDA: 2-3 trades/day, momentum-driven
- TSLA: 3-4 trades/day, high volatility capture
- AAPL: 0-1 trades/day, selective mean-reversion
- MSFT: 0-1 trades/day, high-quality setups only

## Troubleshooting

### No Signals Generated
- Check if market is open
- Verify data download succeeded
- Ensure sufficient historical data (90 days)

### Position Size Zero
- Signal confidence too low
- Account value insufficient
- Risk parameters too conservative

### Workflow Fails
- Check TA-Lib installation
- Verify Python dependencies
- Ensure Azure credentials are set

## Future Enhancements

1. **Additional Tickers**: Expand to more symbols
2. **Options Integration**: Add options strategies
3. **Performance Tracking**: Automated P&L monitoring
4. **Alert System**: Push notifications for high-confidence signals
5. **Machine Learning**: Adaptive model selection based on performance