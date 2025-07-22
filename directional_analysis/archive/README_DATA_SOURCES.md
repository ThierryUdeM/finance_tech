# Data Source Analysis for Pattern Matching

## Current Situation

### Yahoo Finance Limitations
- **Intraday data**: Limited to 60 days lookback
- **Pattern library size**: Only ~100-200 patterns (too small for reliable predictions)
- **No access to TSX data**: AC.TO not available with good intraday history

### Databento Results
- Found "AC" on NASDAQ (XNAS.ITCH dataset)
- Downloaded 1 year of data: Only 8,996 trades → 761 fifteen-minute bars
- **Average 3.7 bars per day** - extremely illiquid
- Cost: Only $0.03 for 1 year (very cheap because so little data)
- **Conclusion**: This is likely an ADR or secondary listing, not suitable for pattern matching

## Recommendations

### 1. For AC.TO Specifically

#### Option A: Use Daily Bars (Recommended)
```python
# Modify the pattern matcher to use daily bars
# Yahoo Finance has years of daily data for AC.TO
ticker = "AC.TO"
interval = "1d"
period = "5y"  # 5 years of daily bars
```

#### Option B: Alternative Data Providers
- **Polygon.io**: $299/month includes TSX real-time and historical
- **Alpha Vantage**: Free tier has 15-min data but limited API calls
- **Interactive Brokers API**: Requires brokerage account
- **Questrade API**: Canadian broker, good for Canadian stocks

### 2. Use More Liquid Tickers

Instead of AC.TO, consider these liquid Canadian stocks:
- **TD.TO** (TD Bank)
- **RY.TO** (Royal Bank)
- **CNR.TO** (Canadian National Railway)
- **SU.TO** (Suncor Energy)

### 3. Hybrid Approach

Combine multiple timeframes:
```python
# Use daily patterns for trend
daily_patterns = analyze_daily("AC.TO", years=5)

# Use recent intraday for timing
intraday_signals = analyze_intraday("AC.TO", days=60)

# Weight the predictions
final_prediction = 0.7 * daily_patterns + 0.3 * intraday_signals
```

### 4. Technical Indicators Approach

Instead of pure pattern matching, use technical indicators:
- RSI, MACD, Bollinger Bands
- Volume patterns
- Support/resistance levels
- More reliable with limited data

## Code Example: Daily Pattern Matcher

```python
# Modified to use daily bars with years of history
def forecast_shape_daily(ticker, period="5y", query_length=20, K=50):
    """
    Use daily bars for better pattern library
    """
    df = yf.download(ticker, period=period, interval="1d")
    
    # Build library with 20-day patterns
    library = build_daily_patterns(df, query_length)
    
    # Much larger library: 1000+ patterns
    print(f"Library size: {len(library)} patterns")
    
    # Make predictions for next 1, 5, 20 days
    return predict_daily_returns(df, library, K)
```

## Conclusion

The 60-day Yahoo Finance limit severely restricts pattern matching effectiveness. For Canadian stocks like AC.TO:

1. **Best free option**: Use daily bars (5+ years available)
2. **Best paid option**: Polygon.io for full TSX historical data
3. **Most practical**: Switch to more liquid tickers or use technical indicators

The Databento AC data we downloaded is too sparse for reliable pattern matching. Consider using it only for testing/development, not actual trading signals.