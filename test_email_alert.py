#!/usr/bin/env python3
"""
Test script for email alert functionality
Tests the email sending capability with a sample trading alert
"""

import os
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def test_email_credentials():
    """Test if email credentials are available"""
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_pwd = os.environ.get('GMAIL_APP_PWD')
    alert_to = os.environ.get('ALERT_TO')
    
    print("Checking email credentials...")
    print(f"GMAIL_USER: {'✓ Set' if gmail_user else '✗ Not set'}")
    print(f"GMAIL_APP_PWD: {'✓ Set' if gmail_pwd else '✗ Not set'}")
    print(f"ALERT_TO: {'✓ Set' if alert_to else '✗ Not set'}")
    
    if not all([gmail_user, gmail_pwd, alert_to]):
        print("\n❌ Missing email credentials. Please set all environment variables.")
        return False
    
    print(f"\nSending from: {gmail_user}")
    print(f"Sending to: {alert_to}")
    return True

def send_test_email():
    """Send a test email alert"""
    gmail_user = os.environ.get('GMAIL_USER')
    gmail_pwd = os.environ.get('GMAIL_APP_PWD')
    alert_to = os.environ.get('ALERT_TO')
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_user
        msg['To'] = alert_to
        msg['Subject'] = "🧪 Test Alert: Trading Pattern Detection System"
        
        # Create test HTML body
        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
                <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
                    <h2 style="color: #333;">Trading Pattern Alert Test</h2>
                    <p>This is a test email to verify the trading alert system is working correctly.</p>
                    
                    <h3 style="color: #555;">Sample Pattern Alert</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Pattern:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">Golden Cross</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Ticker:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">AAPL</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Daily Trigger:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">EMA20 crossed above EMA50, RSI > 55</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Hourly Cue:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;">Price holding above EMA24 with volume spike</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Action:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: #28a745;"><strong>BUY</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Risk Level:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: #28a745;">LOW</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Confidence:</strong></td>
                            <td style="padding: 8px; border-bottom: 1px solid #ddd; color: #28a745;"><strong>HIGH</strong></td>
                        </tr>
                    </table>
                    
                    <div style="margin-top: 20px; padding: 15px; background-color: #d4edda; border-radius: 5px;">
                        <p style="margin: 0; color: #155724;">
                            <strong>✅ Email Alert System Working!</strong><br>
                            You will receive similar alerts when trading patterns are detected.
                        </p>
                    </div>
                    
                    <hr style="margin: 20px 0; border: none; border-top: 1px solid #ddd;">
                    <p style="font-size: 12px; color: #666;">
                        Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EST<br>
                        This is a test message from the Trading Pattern Detection System
                    </p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        # Connect to Gmail and send
        print("\nConnecting to Gmail SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        print("Authenticating...")
        server.login(gmail_user, gmail_pwd)
        
        print("Sending test email...")
        server.send_message(msg)
        server.quit()
        
        print("\n✅ Test email sent successfully!")
        print(f"Check {alert_to} for the test message.")
        return True
        
    except smtplib.SMTPAuthenticationError:
        print("\n❌ Authentication failed!")
        print("Possible issues:")
        print("1. Check that GMAIL_APP_PWD is an app-specific password, not your regular password")
        print("2. Enable 2-factor authentication on your Google account")
        print("3. Generate an app password at: https://myaccount.google.com/apppasswords")
        return False
        
    except smtplib.SMTPException as e:
        print(f"\n❌ SMTP error: {str(e)}")
        return False
        
    except Exception as e:
        print(f"\n❌ Error sending email: {str(e)}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("Trading Alert Email System Test")
    print("="*60)
    
    # Test credentials
    if not test_email_credentials():
        sys.exit(1)
    
    # Ask for confirmation
    print("\nThis will send a test email to verify the alert system.")
    response = input("Do you want to proceed? (y/n): ")
    
    if response.lower() != 'y':
        print("Test cancelled.")
        sys.exit(0)
    
    # Send test email
    if send_test_email():
        print("\n✅ Email system test passed!")
        print("\nNext steps:")
        print("1. Verify you received the test email")
        print("2. Check that the formatting looks correct")
        print("3. The system is ready to send trading alerts")
    else:
        print("\n❌ Email system test failed!")
        print("\nTroubleshooting steps:")
        print("1. Verify all environment variables are set correctly")
        print("2. Ensure you're using an app-specific password for Gmail")
        print("3. Check that 'Less secure app access' is not required (use app password instead)")
        print("4. Verify the sender email has 2-factor authentication enabled")

if __name__ == "__main__":
    main()