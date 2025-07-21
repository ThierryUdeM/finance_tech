#!/usr/bin/env python3
import os
import json
import pandas as pd
from datetime import datetime
from azure.storage.blob import BlobServiceClient
import glob

# Azure connection
account_name = os.environ.get('AZURE_STORAGE_ACCOUNT')
account_key = os.environ.get('AZURE_STORAGE_KEY')
container_name = os.environ.get('AZURE_CONTAINER_NAME', 'finance')  # Default to 'finance' container

if not (account_name and account_key):
    print('Azure credentials not found. Results will only be saved locally.')
    exit(0)

# Create connection
connection_string = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'
blob_service = BlobServiceClient.from_connection_string(connection_string)

# Create container client
container_client = blob_service.get_container_client(container_name)

# Ensure container exists
try:
    container_client.get_container_properties()
except:
    print(f"Creating container: {container_name}")
    container_client.create_container()

# Function to upload file to blob
def upload_to_blob(local_path, blob_path):
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_path)
    with open(local_path, 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"Uploaded: {blob_path}")

try:
    # Find prediction history JSON file
    prediction_files = glob.glob('ChartScanAI_Shiny/evaluation_results/nvda_prediction_history.json')
    if not prediction_files:
        prediction_files = glob.glob('evaluation_results/nvda_prediction_history.json')
    
    if prediction_files:
        # Load prediction history
        with open(prediction_files[0], 'r') as f:
            predictions = json.load(f)
        
        if predictions:
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            
            # Save predictions as CSV
            pred_csv_path = 'nvda_predictions_temp.csv'
            pred_df.to_csv(pred_csv_path, index=False)
            
            # Upload predictions to Azure
            pred_blob_name = f'directional_analysis/nvda_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            upload_to_blob(pred_csv_path, pred_blob_name)
            
            # Also save the latest predictions file (overwrite)
            latest_pred_blob = 'directional_analysis/nvda_predictions_latest.csv'
            upload_to_blob(pred_csv_path, latest_pred_blob)
            
            # Clean up temp file
            os.remove(pred_csv_path)
    
    # Find evaluation metrics files
    metrics_files = glob.glob('ChartScanAI_Shiny/evaluation_results/nvda_metrics_*.json')
    if not metrics_files:
        metrics_files = glob.glob('evaluation_results/nvda_metrics_*.json')
    
    if metrics_files:
        # Get the most recent metrics file
        latest_metrics = max(metrics_files, key=os.path.getctime)
        
        with open(latest_metrics, 'r') as f:
            metrics = json.load(f)
        
        # Convert metrics to a flat structure for CSV
        eval_data = []
        for timeframe in ['1h', '3h', 'eod']:
            if timeframe in metrics:
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'timeframe': timeframe,
                    'total_predictions': metrics[timeframe]['total_predictions'],
                    'correct_predictions': metrics[timeframe]['correct_predictions'],
                    'direction_accuracy': metrics[timeframe]['direction_accuracy'],
                    'avg_error': metrics[timeframe]['avg_error'],
                    'min_error': metrics[timeframe]['min_error'],
                    'max_error': metrics[timeframe]['max_error']
                }
                eval_data.append(row)
        
        if eval_data:
            # Create evaluation DataFrame
            eval_df = pd.DataFrame(eval_data)
            
            # Save evaluation as CSV
            eval_csv_path = 'nvda_evaluation_temp.csv'
            eval_df.to_csv(eval_csv_path, index=False)
            
            # Upload evaluation to Azure
            eval_blob_name = f'directional_analysis/nvda_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            upload_to_blob(eval_csv_path, eval_blob_name)
            
            # Also save the latest evaluation file (overwrite)
            latest_eval_blob = 'directional_analysis/nvda_evaluation_latest.csv'
            upload_to_blob(eval_csv_path, latest_eval_blob)
            
            # Clean up temp file
            os.remove(eval_csv_path)
    
    # Find and upload performance reports
    report_files = glob.glob('ChartScanAI_Shiny/evaluation_results/nvda_performance_*.md')
    if not report_files:
        report_files = glob.glob('evaluation_results/nvda_performance_*.md')
    
    for report in report_files:
        filename = os.path.basename(report)
        blob_name = f'directional_analysis/reports/{filename}'
        upload_to_blob(report, blob_name)
    
    print(f"\nSuccessfully uploaded results to Azure container: {container_name}/directional_analysis/")
    print("Files uploaded:")
    print("- nvda_predictions_[timestamp].csv")
    print("- nvda_predictions_latest.csv")
    print("- nvda_evaluation_[timestamp].csv")
    print("- nvda_evaluation_latest.csv")
    print("- reports/*.md")
    
except Exception as e:
    print(f"Error uploading to Azure: {e}")
    print("Results are saved locally only")