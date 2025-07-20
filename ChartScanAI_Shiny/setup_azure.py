#!/usr/bin/env python3
"""
Setup script to test Azure connection and create initial structure
"""

import os
import sys
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

def test_azure_connection():
    """Test Azure connection and setup container structure"""
    # Load environment variables
    load_dotenv('config/.env')
    
    storage_account_name = os.getenv('STORAGE_ACCOUNT_NAME')
    access_key = os.getenv('ACCESS_KEY')
    container_name = os.getenv('CONTAINER_NAME')
    
    if not all([storage_account_name, access_key, container_name]):
        print("ERROR: Azure credentials not found in config/.env")
        print("\nPlease add the following to your config/.env file:")
        print("STORAGE_ACCOUNT_NAME=your_storage_account")
        print("ACCESS_KEY=your_access_key")
        print("CONTAINER_NAME=your_container_name")
        return False
    
    # Create connection string from components
    connection_string = (
        f"DefaultEndpointsProtocol=https;"
        f"AccountName={storage_account_name};"
        f"AccountKey={access_key};"
        f"EndpointSuffix=core.windows.net"
    )
    
    try:
        # Connect to Azure
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Create or verify container
        try:
            blob_service_client.create_container(container_name)
            print(f"✓ Created container: {container_name}")
        except Exception as e:
            if "ContainerAlreadyExists" in str(e):
                print(f"✓ Container already exists: {container_name}")
            else:
                print(f"✗ Error with container: {e}")
                return False
        
        # Create folder structure by uploading marker files
        folders = [
            'predictions/',
            'evaluations/',
            'charts/',
            'reports/'
        ]
        
        container_client = blob_service_client.get_container_client(container_name)
        
        for folder in folders:
            blob_name = f"{folder}.marker"
            blob_client = container_client.get_blob_client(blob_name)
            try:
                blob_client.upload_blob("Folder marker", overwrite=True)
                print(f"✓ Created folder structure: {folder}")
            except Exception as e:
                print(f"✗ Error creating {folder}: {e}")
        
        print("\n✅ Azure setup complete!")
        print(f"Container: {container_name}")
        print("\nFolder structure:")
        print("  - predictions/  (hourly prediction results)")
        print("  - evaluations/  (performance evaluations)")
        print("  - charts/       (generated charts)")
        print("  - reports/      (weekly performance reports)")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Azure Storage connection...\n")
    
    if test_azure_connection():
        print("\n🎉 Ready to use with GitHub Actions!")
        print("\nNext steps:")
        print("1. Add AZURE_STORAGE_CONNECTION_STRING to GitHub Secrets")
        print("2. Upload YOLO weights to a accessible location")
        print("3. Push changes to trigger the workflow")
    else:
        print("\n❌ Setup failed. Please check your configuration.")
        sys.exit(1)