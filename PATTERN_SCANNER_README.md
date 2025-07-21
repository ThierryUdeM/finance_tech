# Pattern Scanner GitHub Workflow

This GitHub Actions workflow automatically scans for chart patterns, generates trading signals, and tracks their performance.

## Features

- **Automated Scanning**: Runs every 15 minutes during market hours
- **Signal Storage**: Saves all signals to Azure Blob Storage
- **Performance Tracking**: Evaluates signal outcomes and calculates win rates
- **Multi-Ticker Support**: Can scan multiple stocks
- **Daily Reports**: Generates performance summaries

## Setup Instructions

### 1. GitHub Secrets

Add these secrets to your GitHub repository (Settings → Secrets → Actions):

- `AZURE_STORAGE_ACCOUNT`: Your Azure storage account name
- `AZURE_STORAGE_KEY`: Your Azure storage access key
- `AZURE_CONTAINER_NAME`: Your Azure container name
- `PATTERN_SCANNER_CODE`: Copy the entire content of `/home/thierrygc/script/pattern_scanner.py`

### 2. Azure Storage Structure

The workflow creates this folder structure in your Azure container:

```
signals/
  NVDA/
    NVDA_signals_20250721_143000.json
    NVDA_signals_20250721_144500.json
  AAPL/
    AAPL_signals_20250721_143000.json
evaluations/
  NVDA/
    NVDA_evaluation_20250721_160000.json
```

## Usage

### Automatic Scanning

The workflow runs automatically every 15 minutes during market hours (9:30 AM - 4:00 PM ET, Monday-Friday).

### Manual Triggers

You can manually trigger the workflow from GitHub Actions tab:

1. **Scan for Patterns**:
   - Action: `scan`
   - Ticker: e.g., `NVDA`

2. **Evaluate Performance**:
   - Action: `evaluate`
   - Ticker: e.g., `NVDA`

3. **Both**:
   - Action: `both`
   - Runs both scan and evaluate

## Signal Format

Each signal contains:

```json
{
  "timestamp": "2025-07-21 14:30:00",
  "ticker": "NVDA",
  "pattern": "Double Bottom",
  "action": "BUY",
  "current_price": 172.75,
  "entry_price": 174.71,
  "stop_loss": 170.21,
  "target_price": 173.82,
  "risk_reward": 0.2,
  "distance_to_entry": 1.13,
  "status": "pending"
}
```

## Performance Metrics

The evaluation tracks:

- **Win Rate**: Percentage of signals that hit target vs stop loss
- **Status Breakdown**:
  - `pending`: Not triggered yet
  - `triggered`: Entry hit, position open
  - `hit_target`: Target reached (win)
  - `hit_stop`: Stop loss hit (loss)
  - `expired`: Over 24 hours old without triggering

## Adding More Tickers

To scan additional tickers, modify the workflow:

1. Add to the scheduled scan:
   ```yaml
   python pattern_signal_tracker.py scan --ticker AAPL
   python pattern_signal_tracker.py scan --ticker MSFT
   ```

2. Or create a loop in the workflow

## Viewing Results

1. **GitHub Actions**: Check workflow run logs
2. **Azure Storage Explorer**: Browse saved signals and evaluations
3. **Daily Reports**: Automatic GitHub issues with summaries

## Pattern Detection Parameters

Current settings optimized for intraday (5-minute) data:
- Peak detection distance: 3 bars
- Peak width: 2 bars
- Prominence: 0.1

## Risk Management

- Signals include risk/reward ratios
- Only trade signals with R:R > 1:2
- Monitor win rate trends over time
- Adjust parameters based on performance

## Troubleshooting

1. **No patterns found**: Normal during low volatility
2. **Poor win rates**: May need to adjust entry/exit logic
3. **Azure errors**: Check credentials and container permissions

## Future Enhancements

- [ ] Add more chart patterns
- [ ] Include volume analysis
- [ ] Send alerts via email/Discord
- [ ] Create performance dashboard
- [ ] Backtest parameter optimization