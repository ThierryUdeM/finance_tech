#!/usr/bin/env python3
"""
Create GitHub issue with performance summary
"""

import os
import json
import requests
from datetime import datetime
import pytz
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def create_github_issue():
    logger = setup_logging()
    
    # Get GitHub token
    github_token = os.environ.get('GITHUB_TOKEN')
    if not github_token:
        logger.error("GITHUB_TOKEN not found in environment")
        return
    
    # Find the latest performance summary
    try:
        output_files = [f for f in os.listdir('output') if f.startswith('performance_summary_') and f.endswith('.json')]
        if not output_files:
            logger.warning("No performance summary files found")
            return
        
        latest_file = sorted(output_files)[-1]
        
        with open(f'output/{latest_file}', 'r') as f:
            summary = json.load(f)
            
    except Exception as e:
        logger.error(f"Error reading performance summary: {str(e)}")
        return
    
    # Create issue content
    timestamp = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M ET')
    
    title = f"📊 Multi-Ticker Signal Analysis Performance Report - {timestamp}"
    
    body = f"""## 🤖 Multi-Ticker Signal Analysis Performance Summary

**Generated:** {summary['generated_at']}
**Tickers Analyzed:** {summary['total_tickers']}
**Average Accuracy:** {summary['average_accuracy']}%

### 📈 Performance Distribution
- 🟢 **Excellent (≥50%):** {summary['performance_distribution']['excellent']} tickers
- 🔵 **Good (45-49%):** {summary['performance_distribution']['good']} tickers  
- 🟡 **Fair (40-44%):** {summary['performance_distribution']['fair']} tickers
- 🔴 **Needs Improvement (<40%):** {summary['performance_distribution']['needs_improvement']} tickers

### 📊 Individual Ticker Performance

"""
    
    # Add individual ticker performance
    for ticker_summary in sorted(summary['ticker_summaries'], key=lambda x: x['overall_accuracy'], reverse=True):
        rating_emoji = {
            'EXCELLENT': '🟢',
            'GOOD': '🔵', 
            'FAIR': '🟡',
            'NEEDS_IMPROVEMENT': '🔴',
            'NO_DATA': '⚫'
        }.get(ticker_summary['rating'], '❓')
        
        body += f"**{ticker_summary['ticker']}** {rating_emoji} {ticker_summary['overall_accuracy']}%\n"
        
        if 'data_points' in ticker_summary:
            for horizon in ['1h', '3h', 'eod']:
                if horizon in ticker_summary['data_points']:
                    data = ticker_summary['data_points'][horizon]
                    body += f"  - {horizon.upper()}: {data['accuracy']}% ({data['total_predictions']} predictions)\n"
        body += "\n"
    
    body += f"""
### 🎯 Key Insights

"""
    
    # Add insights based on performance
    avg_acc = summary['average_accuracy']
    if avg_acc >= 50:
        body += "✅ **Excellent Overall Performance** - Signal model is performing well above random baseline\n"
    elif avg_acc >= 45:
        body += "✅ **Good Performance** - Signal model is beating random baseline consistently\n"
    elif avg_acc >= 40:
        body += "⚠️ **Fair Performance** - Signal model is close to acceptable threshold\n"
    else:
        body += "❌ **Performance Alert** - Signal model may need optimization or retraining\n"
    
    # Find best and worst performers
    if summary['ticker_summaries']:
        best_ticker = max(summary['ticker_summaries'], key=lambda x: x['overall_accuracy'])
        worst_ticker = min(summary['ticker_summaries'], key=lambda x: x['overall_accuracy'])
        
        body += f"\n🏆 **Best Performer:** {best_ticker['ticker']} ({best_ticker['overall_accuracy']}%)\n"
        body += f"📉 **Needs Attention:** {worst_ticker['ticker']} ({worst_ticker['overall_accuracy']}%)\n"
    
    body += f"""

### 🔍 Next Steps

Based on this analysis:
"""
    
    if avg_acc < 45:
        body += "- 🔧 **Model Optimization Needed** - Consider parameter tuning or additional training data\n"
    
    excellent_count = summary['performance_distribution']['excellent']
    total_count = summary['total_tickers']
    
    if excellent_count < total_count / 2:
        body += "- 📊 **Expand High-Quality Data** - Focus on obtaining more historical data for underperforming tickers\n"
    
    body += "- 📈 **Continue Monitoring** - Regular evaluation helps maintain model performance\n"
    body += "- 🎯 **Pattern Analysis** - Review which patterns work best for each ticker\n"
    
    body += f"""

---
🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
"""
    
    # Create the issue
    try:
        # Extract repo info from git remote
        import subprocess
        result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            repo_url = result.stdout.strip()
            # Parse GitHub repo from URL
            if 'github.com' in repo_url:
                # Handle both SSH and HTTPS URLs
                if repo_url.startswith('git@github.com:'):
                    repo_path = repo_url.replace('git@github.com:', '').replace('.git', '')
                else:
                    repo_path = repo_url.replace('https://github.com/', '').replace('.git', '')
                
                # Create issue via GitHub API
                api_url = f"https://api.github.com/repos/{repo_path}/issues"
                
                headers = {
                    'Authorization': f'token {github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                }
                
                data = {
                    'title': title,
                    'body': body,
                    'labels': ['performance-report', 'signal-analysis', 'automated']
                }
                
                response = requests.post(api_url, headers=headers, json=data)
                
                if response.status_code == 201:
                    issue_url = response.json()['html_url']
                    logger.info(f"✅ GitHub issue created: {issue_url}")
                else:
                    logger.error(f"Failed to create issue: {response.status_code} - {response.text}")
            else:
                logger.warning("Not a GitHub repository")
        else:
            logger.warning("Could not determine git repository")
            
    except Exception as e:
        logger.error(f"Error creating GitHub issue: {str(e)}")

if __name__ == "__main__":
    create_github_issue()