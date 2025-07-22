# Pattern Scanner Workflow

This folder contains only the essential files needed for the GitHub Actions pattern scanner workflow.

## Files Included

- `.github/workflows/pattern_scanner.yml` - GitHub Actions workflow that runs every 15 minutes during trading hours
- `combined_pattern_scanner_gh.py` - Main scanner using TradingPatternScanner (84.5% accuracy) and TA-Lib
- `pattern_evaluator.py` - End-of-day evaluation of detected patterns
- `pattern_signal_tracker.py` - Tracks and evaluates pattern signals
- `requirements_pattern_scanner.txt` - Python dependencies

## Setup Instructions

1. **Copy this folder to your GitHub repository**

2. **Set up GitHub Secrets** (Settings → Secrets and variables → Actions):
   - `STORAGE_ACCOUNT_NAME` - Your Azure storage account name
   - `ACCESS_KEY` - Your Azure storage account access key
   - `CONTAINER_NAME` - Your Azure container name

3. **Create config/.env file** (for local testing):
   ```
   AZURE_STORAGE_ACCOUNT=your_storage_account
   AZURE_STORAGE_KEY=your_access_key
   AZURE_CONTAINER_NAME=your_container_name
   ```

4. **Push to GitHub** and the workflow will run automatically

## Features

- Detects chart patterns using TradingPatternScanner (84.5% accuracy)
- Detects 60+ candlestick patterns using TA-Lib
- Runs every 15 minutes during market hours
- Saves results to Azure blob storage
- End-of-day evaluation of pattern success rates
- Multi-ticker support (NVDA, AAPL, MSFT, GOOGL, TSLA, SPY, QQQ)

## Requirements

- Python 3.10+
- TradingPatternScanner (installed from GitHub)
- TA-Lib (built from source in workflow)