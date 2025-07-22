# Multi-Ticker YOLO Monitor - Shiny Dashboard

A comprehensive R Shiny dashboard that displays YOLO predictions and performance analytics for multiple tickers (BTC-USD, NVDA, AC.TO) from Azure Storage.

## Features

### 📊 Overview Dashboard
- Quick view of all tickers in one place
- Current prices and recommendations
- Mini performance gauges
- 24-hour recommendation comparison chart

### 📈 Individual Ticker Tabs
Each ticker (BTC-USD, NVDA, AC.TO) has its own dedicated tab with:
- **Current Status**: Price, YOLO recommendation, buy/sell signal counts
- **Timeframe Analysis**: Signals breakdown by interval (15m, 1h, 4h, 1d)
- **Performance Gauge**: Visual representation of prediction accuracy
- **24-Hour History**: Chart showing price, recommendations, and signals over time
- **Accuracy Metrics**: Performance by signal type (BUY, SELL, HOLD)

### 🔄 Compare Tab
- Side-by-side comparison of all tickers
- Performance metrics table
- Accuracy trend visualization

### ⚙️ Settings
- Azure connection status
- Adjustable history hours (6-48)
- Evaluation period settings (1-14 days)

## Prerequisites

1. **R packages** (will be auto-installed):
   - shiny
   - shinydashboard
   - plotly
   - DT
   - jsonlite
   - AzureStor
   - lubridate
   - dplyr
   - tidyr

2. **Azure Storage** with predictions data from the multi-ticker predictor

3. **Configuration file** at `config/.env`:
   ```
   STORAGE_ACCOUNT_NAME=your_storage_account
   ACCESS_KEY=your_access_key
   CONTAINER_NAME=your_container
   ```

## Running the App

### Method 1: Using the provided script
```bash
cd ChartScanAI_Shiny
./run_multi_ticker_app.sh
```

### Method 2: From R/RStudio
```r
setwd("path/to/ChartScanAI_Shiny")
shiny::runApp("multi_ticker_monitor_azure.R")
```

### Method 3: Direct R command
```bash
R -e "shiny::runApp('multi_ticker_monitor_azure.R', launch.browser = TRUE)"
```

## Data Structure Expected

The app expects data in Azure Storage with this structure:
```
predictions/
├── BTC-USD/
│   └── YYYY-MM-DD/
│       └── HH.json
├── NVDA/
│   └── YYYY-MM-DD/
│       └── HH.json
└── AC.TO/
    └── YYYY-MM-DD/
        └── HH.json

evaluations/
├── BTC-USD/
│   └── YYYY-MM-DD/
│       └── HHMMSS.json
├── NVDA/
│   └── YYYY-MM-DD/
│       └── HHMMSS.json
└── AC.TO/
    └── YYYY-MM-DD/
        └── HHMMSS.json
```

## Features in Detail

### Auto-refresh
- Toggle auto-refresh (5-minute intervals)
- Manual refresh button for immediate updates

### Alert System
- Bell icon shows count of STRONG BUY/SELL signals
- Quick identification of significant trading opportunities

### Performance Metrics
- Overall accuracy percentage
- Accuracy by signal type (BUY, SELL, HOLD)
- Total predictions evaluated
- Success/failure breakdown

### Interactive Charts
- Hover for detailed information
- Zoom and pan capabilities
- Export chart images

## Customization

### Adding New Tickers
Edit the `TICKERS` configuration in the app:
```r
TICKERS <- list(
  "BTC-USD" = list(name = "Bitcoin", icon = "bitcoin", color = "orange"),
  "NVDA" = list(name = "NVIDIA", icon = "microchip", color = "green"),
  "AC.TO" = list(name = "Air Canada", icon = "plane", color = "red"),
  # Add new ticker:
  "AAPL" = list(name = "Apple", icon = "apple", color = "blue")
)
```

### Modifying Thresholds
Performance evaluation thresholds are set in the Python predictor configuration.

## Troubleshooting

1. **No data showing**: 
   - Check Azure credentials in config/.env
   - Verify container has prediction data
   - Check timezone settings (app expects UTC data)

2. **Connection errors**:
   - Verify internet connection
   - Check Azure Storage firewall settings
   - Ensure access key is valid

3. **Missing packages**:
   - Run the provided script which auto-installs packages
   - Or install manually: `install.packages(c("shiny", "AzureStor", ...))`

## Performance Tips

- Limit history hours for faster loading
- Use manual refresh during active trading
- Close unused tabs to reduce memory usage

## Support

For issues or questions:
1. Check Azure Storage for data availability
2. Verify credentials are correct
3. Check R console for error messages
4. Ensure all R packages are up to date