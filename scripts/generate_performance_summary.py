#!/usr/bin/env python3
"""
Multi-Ticker Signal Analysis - Performance Summary Generator
Generates a comprehensive performance summary across all tickers.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ChartScanAI_Shiny.azure_utils import upload_to_azure, list_azure_blobs

def setup_logging():
    """Set up logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/performance_summary.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_available_tickers():
    """Get list of tickers with available data"""
    tickers = []
    
    # Check for data files
    data_dir = Path('data')
    if data_dir.exists():
        for file_path in data_dir.glob('*_15min_pattern_ready.csv'):
            ticker = file_path.stem.replace('_15min_pattern_ready', '')
            tickers.append(ticker)
    
    return sorted(tickers)

def get_recent_evaluations(ticker, days_back=7):
    """Get recent evaluation results for a ticker"""
    evaluations = []
    
    et_tz = pytz.timezone('US/Eastern')
    current_date = datetime.now(et_tz)
    
    for i in range(days_back):
        check_date = current_date - timedelta(days=i)
        date_str = check_date.strftime('%Y-%m-%d')
        
        # List all evaluation files for this date
        try:
            prefix = f"evaluations/{ticker}/{date_str}/"
            blob_names = list_azure_blobs(prefix)
            
            for blob_name in blob_names:
                if blob_name.endswith('.json'):
                    eval_data = download_from_azure(blob_name)
                    if eval_data:
                        evaluation = json.loads(eval_data)
                        evaluation['source_file'] = blob_name
                        evaluations.append(evaluation)
                        
        except Exception:
            continue
    
    return evaluations

def calculate_ticker_summary(ticker, evaluations, logger):
    """Calculate summary statistics for a ticker"""
    if not evaluations:
        return None
    
    # Get the most recent evaluation with metrics
    recent_eval = None
    for eval_data in sorted(evaluations, key=lambda x: x['evaluation_timestamp'], reverse=True):
        if 'performance_metrics' in eval_data:
            recent_eval = eval_data
            break
    
    if not recent_eval:
        return None
    
    metrics = recent_eval['performance_metrics']
    
    summary = {
        'ticker': ticker,
        'last_evaluation': recent_eval['evaluation_timestamp'],
        'data_points': {}
    }
    
    # Extract key metrics for each horizon
    horizons = ['1h', '3h', 'eod']
    for horizon in horizons:
        accuracy_key = f'{horizon}_accuracy'
        total_key = f'{horizon}_total_predictions'
        error_key = f'{horizon}_avg_error'
        
        if accuracy_key in metrics:
            summary['data_points'][horizon] = {
                'accuracy': metrics[accuracy_key],
                'total_predictions': metrics.get(total_key, 0),
                'avg_error': metrics.get(error_key, 0),
                'bullish_accuracy': metrics.get(f'{horizon}_bullish_accuracy', 0),
                'bearish_accuracy': metrics.get(f'{horizon}_bearish_accuracy', 0),
                'neutral_accuracy': metrics.get(f'{horizon}_neutral_accuracy', 0)
            }
    
    # Calculate overall performance score
    accuracies = [summary['data_points'][h]['accuracy'] for h in horizons 
                 if h in summary['data_points']]
    
    if accuracies:
        summary['overall_accuracy'] = round(sum(accuracies) / len(accuracies), 2)
        summary['best_horizon'] = max(horizons, key=lambda h: summary['data_points'].get(h, {}).get('accuracy', 0))
        
        # Performance rating
        if summary['overall_accuracy'] >= 50:
            summary['rating'] = 'EXCELLENT'
        elif summary['overall_accuracy'] >= 45:
            summary['rating'] = 'GOOD'
        elif summary['overall_accuracy'] >= 40:
            summary['rating'] = 'FAIR'
        else:
            summary['rating'] = 'NEEDS_IMPROVEMENT'
    else:
        summary['overall_accuracy'] = 0
        summary['rating'] = 'NO_DATA'
    
    logger.info(f"{ticker}: {summary['overall_accuracy']}% accuracy ({summary['rating']})")
    
    return summary

def generate_html_report(ticker_summaries, logger):
    """Generate HTML performance report"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Ticker Signal Analysis Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .ticker-summary {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .excellent {{ border-left: 5px solid #28a745; }}
            .good {{ border-left: 5px solid #17a2b8; }}
            .fair {{ border-left: 5px solid #ffc107; }}
            .needs-improvement {{ border-left: 5px solid #dc3545; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            .metrics-table th {{ background-color: #f8f9fa; }}
            .summary-stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
            .stat-box {{ text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🤖 Multi-Ticker Signal Analysis Performance Report</h1>
            <p>Generated: {datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S ET')}</p>
        </div>
    """
    
    if not ticker_summaries:
        html_content += "<p><strong>No performance data available. Run evaluations first.</strong></p>"
    else:
        # Overall statistics
        total_tickers = len(ticker_summaries)
        avg_accuracy = sum(s['overall_accuracy'] for s in ticker_summaries) / total_tickers
        excellent_count = sum(1 for s in ticker_summaries if s['rating'] == 'EXCELLENT')
        good_count = sum(1 for s in ticker_summaries if s['rating'] == 'GOOD')
        
        html_content += f"""
        <div class="summary-stats">
            <div class="stat-box">
                <h3>{total_tickers}</h3>
                <p>Total Tickers</p>
            </div>
            <div class="stat-box">
                <h3>{avg_accuracy:.1f}%</h3>
                <p>Average Accuracy</p>
            </div>
            <div class="stat-box">
                <h3>{excellent_count + good_count}</h3>
                <p>Performing Well</p>
            </div>
        </div>
        """
        
        # Individual ticker summaries
        for summary in sorted(ticker_summaries, key=lambda x: x['overall_accuracy'], reverse=True):
            rating_class = summary['rating'].lower().replace('_', '-')
            
            html_content += f"""
            <div class="ticker-summary {rating_class}">
                <h2>{summary['ticker']} - {summary['overall_accuracy']}% ({summary['rating']})</h2>
                <p><strong>Last Evaluation:</strong> {summary['last_evaluation']}</p>
                <p><strong>Best Horizon:</strong> {summary.get('best_horizon', 'N/A')}</p>
                
                <table class="metrics-table">
                    <tr>
                        <th>Horizon</th>
                        <th>Accuracy</th>
                        <th>Predictions</th>
                        <th>Avg Error</th>
                        <th>Bullish</th>
                        <th>Bearish</th>
                        <th>Neutral</th>
                    </tr>
            """
            
            for horizon in ['1h', '3h', 'eod']:
                if horizon in summary['data_points']:
                    data = summary['data_points'][horizon]
                    html_content += f"""
                    <tr>
                        <td>{horizon.upper()}</td>
                        <td>{data['accuracy']:.1f}%</td>
                        <td>{data['total_predictions']}</td>
                        <td>{data['avg_error']:.2f}%</td>
                        <td>{data['bullish_accuracy']:.1f}%</td>
                        <td>{data['bearish_accuracy']:.1f}%</td>
                        <td>{data['neutral_accuracy']:.1f}%</td>
                    </tr>
                    """
            
            html_content += "</table></div>"
    
    html_content += """
        <div class="header" style="margin-top: 30px;">
            <p><em>🤖 Generated with Claude Code</em></p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def main():
    logger = setup_logging()
    
    try:
        logger.info("Starting performance summary generation")
        
        # Get available tickers
        tickers = get_available_tickers()
        logger.info(f"Found {len(tickers)} tickers with data: {', '.join(tickers)}")
        
        if not tickers:
            logger.warning("No tickers found with historical data")
            return
        
        # Generate summaries for each ticker
        ticker_summaries = []
        
        for ticker in tickers:
            logger.info(f"Processing {ticker}...")
            
            try:
                evaluations = get_recent_evaluations(ticker, days_back=7)
                summary = calculate_ticker_summary(ticker, evaluations, logger)
                
                if summary:
                    ticker_summaries.append(summary)
                else:
                    logger.warning(f"No evaluation data found for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not ticker_summaries:
            logger.warning("No performance summaries generated")
            return
        
        # Generate comprehensive summary
        overall_summary = {
            'generated_at': datetime.now(pytz.timezone('US/Eastern')).isoformat(),
            'total_tickers': len(ticker_summaries),
            'tickers_analyzed': [s['ticker'] for s in ticker_summaries],
            'average_accuracy': round(sum(s['overall_accuracy'] for s in ticker_summaries) / len(ticker_summaries), 2),
            'performance_distribution': {
                'excellent': sum(1 for s in ticker_summaries if s['rating'] == 'EXCELLENT'),
                'good': sum(1 for s in ticker_summaries if s['rating'] == 'GOOD'),
                'fair': sum(1 for s in ticker_summaries if s['rating'] == 'FAIR'),
                'needs_improvement': sum(1 for s in ticker_summaries if s['rating'] == 'NEEDS_IMPROVEMENT')
            },
            'ticker_summaries': ticker_summaries
        }
        
        # Save JSON summary
        os.makedirs('output', exist_ok=True)
        timestamp = datetime.now(pytz.timezone('US/Eastern'))
        json_filename = f"output/performance_summary_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(json_filename, 'w') as f:
            json.dump(overall_summary, f, indent=2, default=str)
        
        logger.info(f"JSON summary saved: {json_filename}")
        
        # Generate HTML report
        html_content = generate_html_report(ticker_summaries, logger)
        html_filename = f"output/performance_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
        
        with open(html_filename, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved: {html_filename}")
        
        # Upload to Azure
        try:
            date_str = timestamp.strftime('%Y-%m-%d')
            time_str = timestamp.strftime('%H%M%S')
            
            # Upload JSON
            json_azure_path = f"reports/performance_summary_{date_str}_{time_str}.json"
            upload_to_azure(json.dumps(overall_summary, default=str), json_azure_path)
            
            # Upload HTML
            html_azure_path = f"reports/performance_report_{date_str}_{time_str}.html"
            upload_to_azure(html_content, html_azure_path)
            
            logger.info("Reports uploaded to Azure")
            
        except Exception as e:
            logger.error(f"Azure upload error: {str(e)}")
        
        # Log summary
        logger.info("📊 Performance Summary:")
        logger.info(f"   Total tickers: {overall_summary['total_tickers']}")
        logger.info(f"   Average accuracy: {overall_summary['average_accuracy']}%")
        logger.info(f"   Performance distribution: {overall_summary['performance_distribution']}")
        
        logger.info("✅ Performance summary generated successfully")
        
    except Exception as e:
        logger.error(f"❌ Error generating performance summary: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()