# Deployment Guide for Trading Signal Pipeline

## GitHub Repository Structure

When deploying to GitHub, use the following structure:

```
your-repo/
├── .github/
│   └── workflows/
│       ├── complete_trading_signals.yml    # Main workflow (required)
│       └── generate_trading_signals.yml    # Alternative workflow (optional)
├── ensemble/
│   ├── ensemble_nvda.py
│   ├── ensemble_tsla.py
│   ├── ensemble_aapl_v2.py
│   ├── ensemble_msft_v2.py
│   ├── ensemble_base.py
│   ├── ensemble_base_v2.py
│   ├── market_regime.py
│   ├── regime_weights.py
│   ├── generate_trading_signals.py
│   ├── position_sizing_calculator.py
│   ├── trading_signal_functions.R
│   ├── README.md
│   └── __init__.py
└── README.md (optional)
```

## Required GitHub Secrets

You must configure these secrets in your GitHub repository settings:

1. **STORAGE_ACCOUNT_NAME** - Your Azure Storage account name
2. **CONTAINER_NAME** - The container name for storing signals (e.g., "trading-signals")
3. **ACCESS_KEY** - Your Azure Storage access key

To add these secrets:
1. Go to your repository on GitHub
2. Click on Settings → Secrets and variables → Actions
3. Click "New repository secret" for each secret

## Workflow Execution

### Automatic Execution
The `complete_trading_signals.yml` workflow runs automatically:
- Every 15 minutes during market hours (9:30 AM - 4:00 PM ET, Monday-Friday)
- Checks if market is open before running
- Uploads results to Azure blob storage

### Manual Execution
To test the workflow:
1. Go to Actions tab in your GitHub repository
2. Select "Complete Trading Signal Pipeline"
3. Click "Run workflow"
4. Click "Run workflow" button

## Azure Storage Structure

The workflow creates the following structure in your Azure container:

```
{CONTAINER_NAME}/
├── master/latest.json              # Combined output
├── model_outputs/latest.json       # Raw model outputs
├── signals/latest.json             # Trading signals
├── summary/latest.json             # Actionable summary
├── sizing/latest.json              # Position sizing
└── history/
    ├── model_outputs/
    ├── signals/
    ├── summary/
    ├── sizing/
    └── master/
```

## Verification Steps

1. **Check Workflow Status**
   - Go to Actions tab
   - Verify workflow runs successfully
   - Check logs for any errors

2. **Verify Azure Upload**
   - Check your Azure storage container
   - Confirm files are being uploaded
   - Verify JSON structure is correct

3. **Test R Shiny Integration**
   - Use the provided R functions to load signals
   - Verify data displays correctly

## Troubleshooting

### Common Issues

1. **Workflow fails with "Azure credentials not set"**
   - Ensure all three secrets are configured correctly
   - Check secret names match exactly

2. **TA-Lib installation fails**
   - This is handled in the workflow
   - No action needed

3. **No signals generated**
   - Check if market is open
   - Verify yfinance can fetch data
   - Check model logs for errors

4. **Path errors**
   - The workflow expects the ensemble folder structure
   - Ensure all files are in correct locations

## Notes

- The main workflow to use is `complete_trading_signals.yml`
- It includes all functionality: model runs, signal generation, and position sizing
- Historical data is preserved with timestamps
- The workflow is idempotent - safe to run multiple times