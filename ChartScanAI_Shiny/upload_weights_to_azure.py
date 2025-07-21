#!/usr/bin/env python3
"""
Upload/Download YOLO weights to/from Azure Storage
Run this once to store the weights in your Azure container
"""

import os
import sys
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

def upload_weights():
    """Upload custom YOLO weights to Azure"""
    load_dotenv('config/.env')
    
    # Azure credentials
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    access_key = os.getenv('ACCESS_KEY')
    container_name = os.getenv('CONTAINER_NAME')
    
    if not all([storage_account_name, access_key, container_name]):
        print("ERROR: Azure credentials not found in config/.env")
        return False
    
    # Weight file path
    weight_file = '../ChartScanAI/weights/custom_yolov8.pt'
    
    if not os.path.exists(weight_file):
        print(f"ERROR: Weight file not found at {weight_file}")
        return False
    
    # Create connection
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={storage_account_name};"
        f"AccountKey={access_key};"
        f"EndpointSuffix=core.windows.net"
    )
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Upload weights
        blob_name = 'weights/custom_yolov8.pt'
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        print(f"Uploading {weight_file} to Azure...")
        with open(weight_file, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
        
        print(f"✓ Successfully uploaded to: {blob_name}")
        print(f"  Container: {container_name}")
        print(f"  Size: {os.path.getsize(weight_file) / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Upload failed: {e}")
        return False

def download_weights():
    """Download YOLO weights from Azure"""
    # Use environment variables (for GitHub Actions)
    storage_account_name = os.getenv('AZURE_STORAGE_ACCOUNT', os.getenv('STORAGE_ACCOUNT_NAME'))
    access_key = os.getenv('AZURE_STORAGE_KEY', os.getenv('ACCESS_KEY'))
    container_name = os.getenv('AZURE_CONTAINER_NAME', os.getenv('CONTAINER_NAME'))
    
    if not all([storage_account_name, access_key, container_name]):
        print("ERROR: Azure credentials not found in environment")
        return False
    
    # Create connection
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={storage_account_name};"
        f"AccountKey={access_key};"
        f"EndpointSuffix=core.windows.net"
    )
    
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create weights directory
        os.makedirs('weights', exist_ok=True)
        
        # Download weights
        blob_name = 'weights/custom_yolov8.pt'
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        weight_file = 'weights/custom_yolov8.pt'
        print(f"Downloading weights from Azure...")
        
        with open(weight_file, 'wb') as f:
            download_stream = blob_client.download_blob()
            f.write(download_stream.readall())
        
        print(f"✓ Successfully downloaded to: {weight_file}")
        print(f"  Size: {os.path.getsize(weight_file) / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        print("Downloading YOLO weights from Azure Storage...\n")
        
        if download_weights():
            print("\n✅ Weights downloaded successfully!")
        else:
            print("\n❌ Download failed.")
    else:
        print("Uploading YOLO weights to Azure Storage...\n")
        
        if upload_weights():
            print("\n✅ Weights uploaded successfully!")
            print("GitHub Actions will now download them automatically.")
        else:
            print("\n❌ Upload failed.")