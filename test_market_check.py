#!/usr/bin/env python3
import datetime
import sys

# Get current time in EST
now_utc = datetime.datetime.utcnow()
now_est = now_utc - datetime.timedelta(hours=5)  # EST is UTC-5

# Check if market is open (9:30 AM - 4:00 PM EST, Mon-Fri)
market_open = datetime.time(9, 30)
market_close = datetime.time(16, 0)

is_weekday = now_est.weekday() < 5
is_market_hours = market_open <= now_est.time() <= market_close

if is_weekday and is_market_hours:
    print('Market is OPEN')
    print(f'Current time EST: {now_est.strftime("%Y-%m-%d %H:%M:%S")}')
    sys.exit(0)
else:
    print('Market is CLOSED')
    print(f'Current time EST: {now_est.strftime("%Y-%m-%d %H:%M:%S")}')
    sys.exit(1)