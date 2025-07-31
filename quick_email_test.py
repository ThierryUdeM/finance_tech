#!/usr/bin/env python3
"""
Quick email test - sends a simple test message
Usage: python quick_email_test.py
"""

import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# Get credentials
gmail_user = os.environ.get('GMAIL_USER')
gmail_pwd = os.environ.get('GMAIL_APP_PWD')
alert_to = os.environ.get('ALERT_TO')

if not all([gmail_user, gmail_pwd, alert_to]):
    print("❌ Missing credentials. Set GMAIL_USER, GMAIL_APP_PWD, and ALERT_TO")
    exit(1)

# Create simple message
msg = MIMEText(f"Test alert sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
msg['Subject'] = '🧪 Quick Test - Trading Alerts'
msg['From'] = gmail_user
msg['To'] = alert_to

# Send
try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(gmail_user, gmail_pwd)
    server.send_message(msg)
    server.quit()
    print(f"✅ Test email sent to {alert_to}")
except Exception as e:
    print(f"❌ Failed: {e}")