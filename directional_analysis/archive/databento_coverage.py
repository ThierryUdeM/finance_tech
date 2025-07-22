#!/usr/bin/env python3
"""
Check Databento's official exchange coverage
"""

import databento as db

API_KEY = "db-C56NjUFsAyQbtQ6en4kkjr4FnKxvb"

def check_coverage():
    """
    Check what Databento officially covers
    """
    
    print("Databento Exchange Coverage")
    print("=" * 60)
    
    print("\nBased on Databento documentation:")
    print("\nUS Equities (included in DBEQ datasets):")
    print("- NYSE (XNYS)")
    print("- NASDAQ (XNAS)")
    print("- NYSE American (XASE)")
    print("- BATS (BATS)")
    print("- IEX (IEXG)")
    
    print("\nFutures/Options:")
    print("- CME Globex (GLBX)")
    print("- OPRA (Options)")
    
    print("\nNOT included in standard datasets:")
    print("- TSX (Toronto Stock Exchange)")
    print("- LSE (London Stock Exchange)")
    print("- Other international exchanges")
    
    print("\n" + "=" * 60)
    print("For TSX data (AC.TO), you need:")
    print("=" * 60)
    
    print("\n1. **Polygon.io** (Recommended)")
    print("   - $299/month for Stocks Developer plan")
    print("   - Includes real-time and historical TSX data")
    print("   - 15-minute delayed data on free tier")
    
    print("\n2. **Alpha Vantage**")
    print("   - Free tier: 5 API calls/minute")
    print("   - Supports TSX with '.TO' suffix")
    print("   - Limited to recent data")
    
    print("\n3. **Yahoo Finance** (Current)")
    print("   - Free but limited to 60 days intraday")
    print("   - Good for daily data (years of history)")
    
    print("\n4. **Interactive Brokers API**")
    print("   - Requires brokerage account")
    print("   - Excellent data quality")
    print("   - $10/month for live data")
    
    print("\n5. **Questrade API**")
    print("   - Canadian broker")
    print("   - Good for Canadian markets")
    print("   - Requires account")
    
    print("\n" + "=" * 60)
    print("RECOMMENDED APPROACH")
    print("=" * 60)
    
    print("\nFor your $120 Databento credit:")
    print("1. Use it for US stocks (better liquidity anyway)")
    print("2. Download 5 years of 15-min data for liquid US stocks")
    print("3. Examples: SPY, QQQ, AAPL, MSFT, etc.")
    
    print("\nFor AC.TO specifically:")
    print("1. Use Polygon.io free tier for testing")
    print("2. Or modify strategy to use daily bars from Yahoo")
    print("3. Or switch to US airline stocks (AAL, DAL, UAL)")

if __name__ == "__main__":
    client = db.Historical(key=API_KEY)
    
    check_coverage()
    
    print("\n\nWould you like to download US airline stocks instead?")
    print("These are available on Databento with good liquidity:")
    print("- AAL (American Airlines)")
    print("- DAL (Delta Air Lines)") 
    print("- UAL (United Airlines)")
    print("- LUV (Southwest Airlines)")