# Multi-Ticker YOLO Predictions - Automated Trading Signal Analysis

Automated trading signal detection for multiple stocks using YOLOv8, running on GitHub Actions with Azure Storage.

## Features

- 🤖 Automated hourly predictions for multiple tickers using custom-trained YOLO
- 📊 Multi-timeframe analysis (15m, 1h, 4h, 1d)
- 💹 Support for BTC-USD, NVDA, and AC.TO
- ☁️ Azure Blob Storage for data persistence
- 📈 Performance tracking and evaluation per ticker
- 📅 Weekly performance reports
- 🔄 Fully automated via GitHub Actions

## Quick Start

1. **Set up GitHub Secrets** (Settings → Secrets → Actions):
   - `STORAGE_ACCOUNT_NAME`
   - `ACCESS_KEY`
   - `CONTAINER_NAME`

2. **Test Azure connection locally**:
   ```bash
   cd ChartScanAI_Shiny
   # Add credentials to config/.env first
   python setup_azure.py
   ```

3. **Push to GitHub**:
   ```bash
   git push origin main
   ```

The workflow will run automatically every hour!

## Project Structure

```
├── .github/workflows/
│   ├── btc-predictions.yml            # Original BTC-only workflow
│   ├── multi-ticker-predictions.yml   # Multi-ticker workflow
│   └── README.md                      # Workflow documentation
├── ChartScanAI_Shiny/
│   ├── btc_predictor_azure.py         # Original BTC prediction script
│   ├── multi_ticker_predictor_azure.py # Multi-ticker prediction script
│   ├── setup_azure.py                 # Azure setup/test script
│   ├── requirements.txt               # Python dependencies
│   └── upload_weights_to_azure.py     # Weight upload utility
└── ChartScanAI/
    └── weights/
        └── custom_yolov8.pt           # Custom-trained YOLO model
```

## How It Works

1. **Every hour**: GitHub Actions triggers the workflow
2. **Data Collection**: Downloads latest price data for all tickers from Yahoo Finance
3. **Chart Generation**: Creates candlestick charts for each ticker and timeframe
4. **YOLO Analysis**: Custom-trained model detects Buy/Sell signals
5. **Azure Storage**: Saves predictions, charts, and evaluations organized by ticker
6. **Performance Tracking**: Evaluates predictions after 1 hour for each ticker
7. **Weekly Reports**: Summarizes accuracy metrics for all tickers

## Supported Tickers

- **BTC-USD**: Bitcoin (Cryptocurrency)
- **NVDA**: NVIDIA Corporation (NASDAQ)
- **AC.TO**: Air Canada (TSX)

## Monitoring Results

View your predictions in Azure Storage Explorer:
- `predictions/{ticker}/` - Hourly prediction results per ticker
- `evaluations/{ticker}/` - Performance tracking per ticker
- `charts/{ticker}/` - Generated candlestick charts per ticker
- `summaries/` - Combined hourly summaries
- `reports/` - Weekly performance summaries

## Local Development

Run predictions locally:
```bash
cd ChartScanAI_Shiny
# For multi-ticker predictions
python multi_ticker_predictor_azure.py

# For BTC-only predictions
python btc_predictor_azure.py
```

## Success Criteria

Success thresholds vary by ticker type:
- **BTC-USD**: Correct if price changes >0.5% within 1 hour
- **NVDA**: Correct if price changes >0.3% within 1 hour  
- **AC.TO**: Correct if price changes >0.5% within 1 hour

## License

This project uses a custom-trained YOLO model specifically for financial chart analysis.