#!/usr/bin/env python3
"""
Performance Dashboard for NVDA Pattern Predictions
Generates HTML dashboard with performance metrics and charts
"""
import pandas as pd
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_performance_history(results_dir='evaluation_results'):
    """Load all historical performance data"""
    history = {
        '1h': [],
        '3h': [],
        'eod': []
    }
    
    metrics_files = list(Path(results_dir).glob('nvda_metrics_*.json'))
    
    for metrics_file in sorted(metrics_files):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
        # Extract timestamp from filename
        timestamp_str = metrics_file.stem.replace('nvda_metrics_', '')
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        
        for timeframe in ['1h', '3h', 'eod']:
            if timeframe in metrics:
                metrics[timeframe]['timestamp'] = timestamp
                history[timeframe].append(metrics[timeframe])
    
    return history

def generate_performance_chart(history, output_dir='performance_charts'):
    """Generate performance trend charts"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NVDA Pattern Prediction Performance Trends', fontsize=16)
    
    # Overall accuracy trend
    ax = axes[0, 0]
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            df = pd.DataFrame(history[tf])
            ax.plot(df['timestamp'], df['direction_accuracy'], marker='o', label=f'{tf.upper()}')
    
    ax.set_title('Direction Accuracy Over Time')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Average error trend
    ax = axes[0, 1]
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            df = pd.DataFrame(history[tf])
            ax.plot(df['timestamp'], df['avg_error'], marker='s', label=f'{tf.upper()}')
    
    ax.set_title('Average Prediction Error Over Time')
    ax.set_ylabel('Error (%)')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Latest accuracy by direction
    ax = axes[1, 0]
    latest_data = []
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            latest = history[tf][-1]
            latest_data.append({
                'Timeframe': tf.upper(),
                'Bullish': latest.get('bullish_accuracy', 0),
                'Bearish': latest.get('bearish_accuracy', 0),
                'Neutral': latest.get('neutral_accuracy', 0)
            })
    
    if latest_data:
        df_latest = pd.DataFrame(latest_data)
        df_latest.set_index('Timeframe').plot(kind='bar', ax=ax)
        ax.set_title('Latest Accuracy by Direction')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xlabel('Timeframe')
        ax.legend(title='Direction')
        ax.grid(True, alpha=0.3, axis='y')
    
    # Prediction count distribution
    ax = axes[1, 1]
    latest_counts = []
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            latest = history[tf][-1]
            latest_counts.append({
                'Timeframe': tf.upper(),
                'Total': latest.get('total_predictions', 0),
                'Bullish': latest.get('bullish_count', 0),
                'Bearish': latest.get('bearish_count', 0),
                'Neutral': latest.get('neutral_count', 0)
            })
    
    if latest_counts:
        df_counts = pd.DataFrame(latest_counts)
        df_counts.set_index('Timeframe')[['Bullish', 'Bearish', 'Neutral']].plot(
            kind='bar', stacked=True, ax=ax
        )
        ax.set_title('Prediction Distribution (Latest)')
        ax.set_ylabel('Count')
        ax.set_xlabel('Timeframe')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'performance_trends.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def generate_html_dashboard(history, chart_path, output_file='performance_dashboard.html'):
    """Generate HTML dashboard"""
    
    # Get latest metrics
    latest_metrics = {}
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            latest_metrics[tf] = history[tf][-1]
    
    # Calculate overall statistics
    overall_accuracy = sum(latest_metrics[tf]['direction_accuracy'] 
                          for tf in latest_metrics) / len(latest_metrics) if latest_metrics else 0
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>NVDA Pattern Prediction Performance Dashboard</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
            color: #495057;
        }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .poor {{ color: #dc3545; }}
        .chart-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .timestamp {{
            text-align: center;
            color: #6c757d;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>NVDA Pattern Prediction Performance Dashboard</h1>
        
        <div class="timestamp">
            Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        
        <div class="summary">
            <div class="metric-card">
                <h3>Overall Accuracy</h3>
                <div class="metric-value {get_performance_class(overall_accuracy)}">{overall_accuracy:.1f}%</div>
                <div class="metric-label">Average across all timeframes</div>
            </div>
    """
    
    # Add metrics for each timeframe
    for tf in ['1h', '3h', 'eod']:
        if tf in latest_metrics:
            m = latest_metrics[tf]
            html_content += f"""
            <div class="metric-card">
                <h3>{tf.upper()} Accuracy</h3>
                <div class="metric-value {get_performance_class(m['direction_accuracy'])}">{m['direction_accuracy']}%</div>
                <div class="metric-label">{m['total_predictions']} predictions evaluated</div>
            </div>
            """
    
    html_content += """
        </div>
        
        <h2>Detailed Performance Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Timeframe</th>
                    <th>Direction Accuracy</th>
                    <th>Avg Error</th>
                    <th>Bullish Acc.</th>
                    <th>Bearish Acc.</th>
                    <th>Neutral Acc.</th>
                    <th>Total Predictions</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for tf in ['1h', '3h', 'eod']:
        if tf in latest_metrics:
            m = latest_metrics[tf]
            html_content += f"""
                <tr>
                    <td><strong>{tf.upper()}</strong></td>
                    <td class="{get_performance_class(m['direction_accuracy'])}">{m['direction_accuracy']}%</td>
                    <td>{m['avg_error']:.3f}%</td>
                    <td>{m.get('bullish_accuracy', 0):.1f}%</td>
                    <td>{m.get('bearish_accuracy', 0):.1f}%</td>
                    <td>{m.get('neutral_accuracy', 0):.1f}%</td>
                    <td>{m['total_predictions']}</td>
                </tr>
            """
    
    html_content += f"""
            </tbody>
        </table>
        
        <div class="chart-container">
            <h2>Performance Trends</h2>
            <img src="{os.path.basename(chart_path)}" alt="Performance Trends">
        </div>
        
        <h2>Analysis Summary</h2>
        <ul>
            <li><strong>Best Performing Timeframe:</strong> {get_best_timeframe(latest_metrics)}</li>
            <li><strong>Most Accurate Direction:</strong> {get_most_accurate_direction(latest_metrics)}</li>
            <li><strong>Performance Trend:</strong> {get_trend_analysis(history)}</li>
        </ul>
    </div>
</body>
</html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    return output_file

def get_performance_class(accuracy):
    """Get CSS class based on accuracy"""
    if accuracy >= 55:
        return 'good'
    elif accuracy >= 45:
        return 'warning'
    else:
        return 'poor'

def get_best_timeframe(metrics):
    """Identify best performing timeframe"""
    if not metrics:
        return "No data available"
    
    best_tf = max(metrics.items(), key=lambda x: x[1]['direction_accuracy'])
    return f"{best_tf[0].upper()} ({best_tf[1]['direction_accuracy']}% accuracy)"

def get_most_accurate_direction(metrics):
    """Identify most accurate direction prediction"""
    all_accuracies = []
    
    for tf, m in metrics.items():
        all_accuracies.extend([
            ('Bullish', m.get('bullish_accuracy', 0)),
            ('Bearish', m.get('bearish_accuracy', 0)),
            ('Neutral', m.get('neutral_accuracy', 0))
        ])
    
    if all_accuracies:
        best = max(all_accuracies, key=lambda x: x[1])
        return f"{best[0]} ({best[1]:.1f}% accuracy)"
    
    return "No data available"

def get_trend_analysis(history):
    """Analyze performance trend"""
    trends = []
    
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) >= 2:
            recent = history[tf][-5:]  # Last 5 evaluations
            if len(recent) >= 2:
                first_acc = recent[0]['direction_accuracy']
                last_acc = recent[-1]['direction_accuracy']
                
                if last_acc > first_acc + 2:
                    trends.append(f"{tf.upper()} improving")
                elif last_acc < first_acc - 2:
                    trends.append(f"{tf.upper()} declining")
    
    return ', '.join(trends) if trends else "Stable performance"

if __name__ == "__main__":
    # Check if evaluation results exist
    if not os.path.exists('evaluation_results'):
        print("No evaluation results found. Run evaluate_nvda_patterns.py first.")
        exit(1)
    
    # Load performance history
    history = load_performance_history()
    
    # Generate charts
    chart_path = generate_performance_chart(history)
    
    # Generate HTML dashboard
    dashboard_path = generate_html_dashboard(history, chart_path)
    
    print(f"Performance dashboard generated: {dashboard_path}")
    print(f"Performance chart saved: {chart_path}")
    
    # Calculate and print summary
    overall_accuracies = []
    for tf in ['1h', '3h', 'eod']:
        if len(history[tf]) > 0:
            latest = history[tf][-1]
            overall_accuracies.append(latest['direction_accuracy'])
    
    if overall_accuracies:
        print(f"\nLatest Overall Accuracy: {sum(overall_accuracies)/len(overall_accuracies):.2f}%")