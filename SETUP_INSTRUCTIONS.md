# Pattern Scanner Setup Instructions

## Step 1: Prepare the Pattern Scanner Code

1. Copy the content of `/home/thierrygc/script/pattern_scanner.py`
2. Save it somewhere temporarily (you'll need it for GitHub secrets)

## Step 2: GitHub Repository Setup

1. Push this folder to your GitHub repository:
   ```bash
   cd /home/thierrygc/test_1/github_ready
   git add .
   git commit -m "Add pattern scanner workflow"
   git push
   ```

## Step 3: Add GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:

1. **AZURE_STORAGE_ACCOUNT**: Your Azure storage account name
2. **AZURE_STORAGE_KEY**: Your Azure storage access key  
3. **AZURE_CONTAINER_NAME**: Your Azure container name
4. **PATTERN_SCANNER_CODE**: 
   - Click "New repository secret"
   - Name: `PATTERN_SCANNER_CODE`
   - Value: Paste the entire content of pattern_scanner.py

## Step 4: Test Locally First

1. Create config/.env file:
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your Azure credentials
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements_pattern_scanner.txt
   ```

3. Run test:
   ```bash
   python test_pattern_tracker.py
   ```

## Step 5: Enable GitHub Actions

1. Go to Actions tab in your repository
2. Enable workflows if not already enabled
3. Find "Pattern Scanner and Evaluator"
4. Click "Run workflow" to test manually

## Step 6: Monitor Performance

The workflow will:
- Run every 15 minutes during market hours
- Save signals to Azure: `signals/TICKER/TICKER_signals_TIMESTAMP.json`
- Evaluate performance and save to: `evaluations/TICKER/TICKER_evaluation_TIMESTAMP.json`

## Workflow Schedule

- **Automatic**: Every 15 minutes, Monday-Friday, 9:30 AM - 4:00 PM ET
- **Manual**: Anytime via GitHub Actions tab

## Viewing Results

1. **Real-time logs**: GitHub Actions run logs
2. **Signal history**: Azure Storage Explorer
3. **Win rates**: Check evaluation files in Azure

## Customization

To add more tickers, edit the workflow file:
- Add to the scan job
- Include in daily report

## Troubleshooting

### Azure Connection Issues
- Verify secret names match exactly
- Check Azure firewall settings
- Ensure container exists

### No Patterns Found
- Normal during low volatility
- Check if market is open
- Verify ticker symbol

### Import Errors
- Ensure PATTERN_SCANNER_CODE secret contains full code
- Check for proper indentation