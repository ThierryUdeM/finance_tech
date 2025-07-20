# BTC YOLO Predictions - GitHub Actions

This workflow automatically runs Bitcoin price predictions using YOLO every hour and stores results in Azure Blob Storage.

## Features

- **Hourly Predictions**: Runs every hour at minute 0
- **Multi-timeframe Analysis**: 15m, 1h, 4h, 1d charts
- **Azure Storage**: All data stored in your Azure container
- **Performance Tracking**: Evaluates predictions after 1 hour
- **Weekly Reports**: Generates performance reports every Sunday

## Setup

### 1. Azure Storage Setup

1. Create an Azure Storage Account
2. Get your credentials from Azure Portal:
   - Storage account name
   - Access key (from Access keys section)
   - Choose a container name
3. Add to `config/.env` for local testing:
   ```
   STORAGE_ACCOUNT_NAME=your_storage_account
   ACCESS_KEY=your_access_key
   CONTAINER_NAME=your_container_name
   ```
4. Test connection:
   ```bash
   cd ChartScanAI_Shiny
   python setup_azure.py
   ```

### 2. GitHub Secrets

Add the following secrets to your repository:
- Go to Settings → Secrets and variables → Actions
- Add these secrets:
  - `STORAGE_ACCOUNT_NAME`: Your Azure storage account name
  - `ACCESS_KEY`: Your Azure storage account access key
  - `CONTAINER_NAME`: The container name for storing predictions

### 3. YOLO Weights

The workflow needs access to the YOLO weights file. Options:
1. Include in repository (if small enough)
2. Host on Azure Blob Storage and download in workflow
3. Use GitHub Releases to store the weights

### 4. Enable GitHub Actions

Push the workflow file to trigger it:
```bash
git add .github/workflows/btc-predictions.yml
git commit -m "Add BTC prediction workflow"
git push
```

## Workflow Schedule

- **Predictions**: Every hour at :00
- **Weekly Report**: Sundays at midnight UTC
- **Manual Run**: Use "Run workflow" button in Actions tab

## Azure Storage Structure

```
btc-predictions/
├── predictions/
│   └── YYYY-MM-DD/
│       └── HH.json          # Hourly predictions
├── evaluations/
│   └── YYYY-MM-DD/
│       └── HHMMSS.json      # Performance evaluations
├── charts/
│   └── YYYY-MM-DD/
│       └── HH/
│           ├── 15m.png
│           ├── 1h.png
│           ├── 4h.png
│           └── 1d.png
└── reports/
    └── performance_YYYYMMDD_HHMMSS.json
```

## Monitoring

### View Latest Predictions
Check Azure Storage Explorer or use Azure CLI:
```bash
az storage blob list --container-name btc-predictions --prefix predictions/
```

### Check Performance
Weekly reports show:
- Overall accuracy percentage
- Buy/Sell/Hold signal accuracy
- Total predictions evaluated

### GitHub Actions Summary
Each run creates a summary with:
- Current recommendation
- Number of buy/sell signals
- Timestamp

## Customization

### Change Schedule
Edit `.github/workflows/btc-predictions.yml`:
```yaml
schedule:
  - cron: '0 */2 * * *'  # Every 2 hours
```

### Adjust Evaluation Criteria
Edit `btc_predictor_azure.py`:
```python
CONFIG = {
    'evaluation_threshold': 1.0  # Require 1% price change
}
```

### Add More Tickers
Modify the script to analyze multiple cryptocurrencies:
```python
CONFIG = {
    'tickers': ['BTC-USD', 'ETH-USD', 'SOL-USD']
}
```

## Troubleshooting

1. **Workflow not running**: Check Actions tab for errors
2. **Azure connection failed**: Verify secret is set correctly
3. **YOLO model not found**: Check weights path in workflow
4. **No evaluations**: Wait at least 1 hour after first run

## Costs

- **GitHub Actions**: Free for public repos (2000 minutes/month)
- **Azure Storage**: Minimal (~$0.01/month for this usage)

## Local Testing

Run the predictor locally:
```bash
cd ChartScanAI_Shiny
python btc_predictor_azure.py
```