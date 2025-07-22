# YOLOv8 Chart Analysis

This folder contains the YOLOv8-based technical analysis system that detects chart patterns and generates trading signals.

## Structure

- **ChartScanAI/** - Core YOLOv8 model and weights
  - `weights/custom_yolov8.pt` - Trained model weights
  
- **ChartScanAI_Shiny/** - Analysis scripts and Shiny app
  - `multi_ticker_predictor_azure.py` - Original multi-ticker predictor
  - `multi_ticker_predictor_azure_optimized.py` - Optimized version with consolidated storage
  - `btc_predictor_azure.py` - Bitcoin-specific predictor
  - Other supporting scripts

- **.github/workflows/** - GitHub Actions workflows
  - `multi-ticker-predictions.yml` - Runs hourly predictions for BTC, NVDA, AC.TO
  - `btc-predictions.yml` - Bitcoin-only predictions (consider disabling)

## Setup

1. Ensure Azure credentials are set in GitHub Secrets:
   - `STORAGE_ACCOUNT_NAME`
   - `ACCESS_KEY`
   - `CONTAINER_NAME`

2. The workflows will run automatically on schedule or can be triggered manually

## Optimization

The optimized version (`multi_ticker_predictor_azure_optimized.py`) consolidates storage to reduce Azure operations:
- All predictions in one file: `consolidated/predictions_latest.json`
- All evaluations in one file: `consolidated/evaluations_latest.json`
- Automatic cleanup of data older than 7 days

To switch to the optimized version, update the workflow to run `multi_ticker_predictor_azure_optimized.py` instead.