# BTC YOLO Predictions - Automated Trading Signal Analysis

Automated Bitcoin trading signal detection using YOLOv8, running on GitHub Actions with Azure Storage.

## Features

- 🤖 Automated hourly BTC price predictions using custom-trained YOLO
- 📊 Multi-timeframe analysis (15m, 1h, 4h, 1d)
- ☁️ Azure Blob Storage for data persistence
- 📈 Performance tracking and evaluation
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
│   ├── btc-predictions.yml    # GitHub Actions workflow
│   └── README.md              # Workflow documentation
├── ChartScanAI_Shiny/
│   ├── btc_predictor_azure.py # Main prediction script
│   ├── setup_azure.py         # Azure setup/test script
│   ├── requirements.txt       # Python dependencies
│   └── upload_weights_to_azure.py # Weight upload utility
└── ChartScanAI/
    └── weights/
        └── custom_yolov8.pt   # Custom-trained YOLO model
```

## How It Works

1. **Every hour**: GitHub Actions triggers the workflow
2. **Data Collection**: Downloads latest BTC price data from Yahoo Finance
3. **Chart Generation**: Creates candlestick charts for each timeframe
4. **YOLO Analysis**: Custom-trained model detects Buy/Sell signals
5. **Azure Storage**: Saves predictions, charts, and evaluations
6. **Performance Tracking**: Evaluates predictions after 1 hour
7. **Weekly Reports**: Summarizes accuracy metrics

## Monitoring Results

View your predictions in Azure Storage Explorer:
- `predictions/` - Hourly prediction results
- `evaluations/` - Performance tracking
- `charts/` - Generated candlestick charts
- `reports/` - Weekly performance summaries

## Local Development

Run predictions locally:
```bash
cd ChartScanAI_Shiny
python btc_predictor_azure.py
```

## Success Criteria

- **BUY**: Correct if price increases >0.5% within 1 hour
- **SELL**: Correct if price decreases >0.5% within 1 hour
- **HOLD**: Correct if price stays within ±0.5%

## License

This project uses a custom-trained YOLO model specifically for financial chart analysis.