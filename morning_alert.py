#!/usr/bin/env python3
"""
Morning Alert Script
Retrieves next-day predictions and sends morning trading alerts
"""

import os
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Azure credentials
load_dotenv('config/.env')

class MorningAlert:
    def __init__(self):
        # Azure configuration
        self.storage_account = os.getenv('AZURE_STORAGE_ACCOUNT')
        self.storage_key = os.getenv('AZURE_STORAGE_KEY')
        self.container_name = os.getenv('AZURE_CONTAINER_NAME')
        
        if not all([self.storage_account, self.storage_key, self.container_name]):
            raise ValueError("Azure storage credentials not found")
        
        # Initialize Azure client
        self.blob_service_client = BlobServiceClient(
            account_url=f"https://{self.storage_account}.blob.core.windows.net",
            credential=self.storage_key
        )
    
    def get_predictions(self):
        """Retrieve next day predictions from Azure"""
        blob_name = "next_day_technical/next_day_predictions.json"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            predictions = json.loads(content)
            return predictions
        except Exception as e:
            logger.error(f"Error retrieving predictions: {e}")
            return None
    
    def format_alert_message(self, predictions):
        """Format predictions into alert message"""
        if not predictions:
            return "No predictions available for today."
        
        summary = predictions['summary']
        
        message = f"""
========================================
DAILY TECHNICAL PATTERN TRADING ALERT
========================================
Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p ET')}
Ticker: {predictions['ticker']}
Market Close Date: {predictions['market_close_date']}

RECOMMENDATION: {summary['recommendation']}
Confidence: {summary['confidence']:.1%}

Pattern Summary:
- Total Patterns: {summary['total_patterns']}
- Bullish: {summary['bullish_count']} (strength: {summary.get('bullish_strength', 0):.2f})
- Bearish: {summary['bearish_count']} (strength: {summary.get('bearish_strength', 0):.2f})
- Neutral: {summary['neutral_count']}

"""
        
        if predictions['predictions']:
            message += "Active Patterns:\n"
            for pattern in predictions['predictions'][:5]:  # Show top 5 patterns
                message += f"\n- {pattern['pattern']} ({pattern['signal']})"
                message += f"\n  Detected: {pattern['timestamp']}"
                message += f"\n  Strength: {pattern['strength']:.1%}"
                message += f"\n  Volume Confirmation: {pattern['volume_confirmation']:.1%}"
                message += f"\n  Entry Price: ${pattern.get('entry_price', 'N/A'):.2f}"
                message += f"\n  Hold for: {pattern['holding_days']} days"
                message += "\n"
        
        message += "\n========================================\n"
        message += "Note: This is based on technical pattern analysis.\n"
        message += "Always do your own research and manage risk appropriately.\n"
        
        return message
    
    def send_alert(self):
        """Send morning alert"""
        predictions = self.get_predictions()
        
        if not predictions:
            logger.warning("No predictions found for morning alert")
            return
        
        # Check if predictions are from yesterday
        pred_date = datetime.fromisoformat(predictions['scan_date']).date()
        today = datetime.now().date()
        
        if (today - pred_date).days > 1:
            logger.warning(f"Predictions are stale (from {pred_date}). Skipping alert.")
            return
        
        # Format and display alert
        alert_message = self.format_alert_message(predictions)
        
        # For GitHub Actions, we'll output to logs
        print(alert_message)
        
        # Save alert to file for artifact
        with open('morning_alert.txt', 'w') as f:
            f.write(alert_message)
        
        logger.info(f"Morning alert sent for {predictions['ticker']}: "
                   f"{predictions['summary']['recommendation']}")
        
        return alert_message
    
    def get_evaluation_summary(self):
        """Get recent evaluation summary"""
        blob_name = "next_day_technical/pattern_evaluations.json"
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            blob_data = blob_client.download_blob()
            content = blob_data.readall()
            evaluations = json.loads(content)
            
            if not evaluations:
                return "No evaluation history available."
            
            # Get last 10 evaluations
            recent_evals = evaluations[-10:]
            
            # Calculate performance metrics
            total_patterns = sum(e['patterns_found'] for e in recent_evals)
            buy_signals = sum(1 for e in recent_evals if e['recommendation'] == 'BUY')
            sell_signals = sum(1 for e in recent_evals if e['recommendation'] == 'SELL')
            avg_confidence = sum(e['confidence'] for e in recent_evals) / len(recent_evals)
            
            summary = f"""
Recent Pattern Performance (Last 10 Scans):
- Total Patterns Found: {total_patterns}
- Buy Signals: {buy_signals}
- Sell Signals: {sell_signals}
- Hold Signals: {len(recent_evals) - buy_signals - sell_signals}
- Average Confidence: {avg_confidence:.1%}
"""
            return summary
            
        except Exception as e:
            logger.error(f"Error retrieving evaluation summary: {e}")
            return "Unable to retrieve evaluation summary."


def main():
    """Main function for GitHub Actions"""
    alert = MorningAlert()
    
    # Send morning alert
    alert.send_alert()
    
    # Add evaluation summary
    eval_summary = alert.get_evaluation_summary()
    print(eval_summary)
    
    with open('morning_alert.txt', 'a') as f:
        f.write(eval_summary)
    
    logger.info("Morning alert process completed")
    return 0


if __name__ == "__main__":
    exit(main())