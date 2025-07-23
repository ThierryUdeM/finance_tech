#!/usr/bin/env python3
"""
Azure Storage utilities for signal analysis
"""

import os
import json
from azure.storage.blob import BlobServiceClient
try:
    from dotenv import load_dotenv
    # Try to load .env file if it exists
    if os.path.exists('config/.env'):
        load_dotenv('config/.env')
    elif os.path.exists('.env'):
        load_dotenv('.env')
except ImportError:
    # dotenv not available, rely on environment variables
    pass

def get_blob_service_client():
    """Initialize Azure Blob Service Client"""
    account_name = os.environ.get('AZURE_STORAGE_ACCOUNT') or os.environ.get('STORAGE_ACCOUNT_NAME')
    account_key = os.environ.get('AZURE_STORAGE_KEY') or os.environ.get('ACCESS_KEY')
    
    if not account_name or not account_key:
        raise ValueError("Azure storage credentials not found in environment variables")
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connection_string)

def upload_to_azure(data, blob_path):
    """Upload data to Azure Blob Storage"""
    try:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get('AZURE_CONTAINER_NAME') or os.environ.get('CONTAINER_NAME')
        
        if not container_name:
            raise ValueError("Container name not found in environment variables")
        
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_path
        )
        
        # Upload data (handle both string and bytes)
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        blob_client.upload_blob(data, overwrite=True)
        return True
        
    except Exception as e:
        print(f"Error uploading to Azure: {str(e)}")
        return False

def download_from_azure(blob_path):
    """Download data from Azure Blob Storage"""
    try:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get('AZURE_CONTAINER_NAME') or os.environ.get('CONTAINER_NAME')
        
        if not container_name:
            raise ValueError("Container name not found in environment variables")
        
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_path
        )
        
        # Download data
        data = blob_client.download_blob().readall()
        
        # Try to decode as string
        try:
            return data.decode('utf-8')
        except:
            return data
            
    except Exception as e:
        raise Exception(f"Error downloading from Azure: {str(e)}")

def list_azure_blobs(prefix=None):
    """List blobs in Azure container with optional prefix filter"""
    try:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get('AZURE_CONTAINER_NAME') or os.environ.get('CONTAINER_NAME')
        
        if not container_name:
            raise ValueError("Container name not found in environment variables")
        
        container_client = blob_service_client.get_container_client(container_name)
        
        if prefix:
            blobs = container_client.list_blobs(name_starts_with=prefix)
        else:
            blobs = container_client.list_blobs()
            
        return [blob.name for blob in blobs]
        
    except Exception as e:
        print(f"Error listing Azure blobs: {str(e)}")
        return []

def blob_exists(blob_path):
    """Check if a blob exists in Azure storage"""
    try:
        blob_service_client = get_blob_service_client()
        container_name = os.environ.get('AZURE_CONTAINER_NAME') or os.environ.get('CONTAINER_NAME')
        
        if not container_name:
            raise ValueError("Container name not found in environment variables")
        
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_path
        )
        
        return blob_client.exists()
        
    except Exception as e:
        print(f"Error checking blob existence: {str(e)}")
        return False