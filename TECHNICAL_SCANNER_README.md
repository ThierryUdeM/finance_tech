# Technical Scanner Setup Guide

This folder contains two technical analysis systems optimized for different trading timeframes:

## 1. Daily Pattern Scanner (Swing Trading)
- **Schedule**: Runs at 3:55 PM ET daily
- **Purpose**: Detects candlestick and chart patterns on daily charts
- **Holding Period**: 2-5 days per pattern
- **Azure Folder**: `next_day_technical/`
- **Workflow**: `.github/workflows/pattern_scanner.yml`

### Daily Scanner Features:
- TA-Lib candlestick patterns (Hammer, Doji, Engulfing, etc.)
- TradingPatterns chart patterns (Head & Shoulders, Double Top/Bottom, Triangles)
- Volume confirmation for all patterns
- Next-day entry price calculations
- Morning alert at 9:15 AM ET

## 2. Simple Technical Scanner (Intraday Trading)
- **Schedule**: Every 15 minutes during market hours
- **Purpose**: Momentum-based signals for 15-minute trading
- **Holding Period**: 15 minutes to 2 hours
- **Azure Folder**: `same_day_technical/`
- **Workflow**: `.github/workflows/simple_technical_scanner.yml`

### Intraday Scanner Features:
- Moving average crossovers (10/30 period)
- RSI with dynamic thresholds based on volatility
- Stochastic oscillator (14,3)
- MACD histogram
- Bollinger Bands
- Volume confirmation
- ATR-based stop loss and take profit levels

## Performance Results (Walk-Forward Testing)

Based on 2.5 years of NVDA 15-minute data:

### Simple Technical (Intraday) - RECOMMENDED
- **Return**: +3.66% over 3 weeks
- **Sharpe Ratio**: 2.60
- **Win Rate**: 45.6%
- **Trades**: 147

### Daily Pattern Scanner
- **Return**: -1.29% over 3 weeks
- **Sharpe Ratio**: -0.60
- **Win Rate**: 43.2%
- **Trades**: 340 (overtrading on 15-min data)

**Conclusion**: The simple technical indicators perform much better for intraday trading. The pattern scanner should be used only for daily charts and swing trading.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install yfinance pandas numpy azure-storage-blob python-dotenv
   pip install TA-Lib  # For daily scanner
   pip install tradingpatterns  # For chart patterns
   ```

2. **Configure Azure Secrets** in GitHub:
   - `STORAGE_ACCOUNT_NAME`
   - `ACCESS_KEY`
   - `CONTAINER_NAME`

3. **Deploy Workflows**:
   - Push both workflow files to `.github/workflows/`
   - Workflows will run automatically on schedule
   - Can also trigger manually from GitHub Actions

## Shiny App Integration

The Multi-Ticker Monitor app has been updated with two new sections:

### Next Day Technical Signals
- Shows patterns detected at market close
- Displays entry prices for next trading day
- Includes recommendation (BUY/SELL/HOLD) with confidence

### Intraday Technical Signals
- Real-time momentum signals updated every 15 minutes
- Shows current signal with stop loss and take profit
- Displays all technical indicators and signal components
- Performance metrics for last 100 signals

## Files Structure
```
technical/
├── daily_pattern_scanner.py        # Daily pattern detection
├── simple_technical_scanner.py     # Intraday momentum signals
├── morning_alert.py               # 9:15 AM alert script
├── combined_pattern_scanner_gh.py  # Original pattern scanner (legacy)
├── .github/
│   └── workflows/
│       ├── pattern_scanner.yml    # Daily scanner workflow
│       └── simple_technical_scanner.yml  # Intraday workflow
└── requirements_pattern_scanner.txt
```

## Azure Storage Structure
```
container/
├── next_day_technical/
│   ├── next_day_predictions.json  # Current predictions
│   └── pattern_evaluations.json   # Historical evaluations (append-only)
└── same_day_technical/
    ├── current_signal.json        # Latest 15-min signal
    ├── technical_evaluations.json # Historical signals (append-only)
    └── performance_summary.json   # Performance metrics
```