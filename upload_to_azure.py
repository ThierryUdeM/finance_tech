#!/usr/bin/env python3
import os
from azure.storage.blob import BlobServiceClient
from datetime import datetime

# Azure connection using account key
account_name = os.environ.get('AZURE_STORAGE_ACCOUNT')
account_key = os.environ.get('AZURE_STORAGE_KEY')
container_name = os.environ.get('AZURE_CONTAINER_NAME')

if account_name and account_key and container_name:
    # Create connection string from components
    connection_string = f'DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net'
    
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    
    # Upload the file
    blob_name = f'NVDA/NVDA_15min_pattern_ready_{datetime.now().strftime("%Y%m%d")}.csv'
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
    
    with open('data/NVDA_15min_pattern_ready.csv', 'rb') as data:
        blob_client.upload_blob(data, overwrite=True)
    
    print(f'Uploaded NVDA data to Azure: {blob_name}')
    print(f'Container: {container_name}')
    print(f'Storage Account: {account_name}')
else:
    print('Azure credentials not found. Please set:')
    print('- STORAGE_ACCOUNT_NAME')
    print('- ACCESS_KEY')
    print('- CONTAINER_NAME')
    print('Data saved locally only')