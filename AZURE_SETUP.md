# Azure Blob Storage Setup for GitHub Actions

## Required Secrets

You need to add these three secrets to your GitHub repository:

1. **STORAGE_ACCOUNT_NAME** - Your Azure storage account name
2. **ACCESS_KEY** - Your Azure storage account access key
3. **CONTAINER_NAME** - The container name where data will be stored

## How to Add Secrets

1. Go to your repository: https://github.com/ThierryUdeM/finance_tech
2. Click on **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret:
   - Name: `STORAGE_ACCOUNT_NAME`
   - Value: [Your storage account name]
   - Click "Add secret"
   
   Repeat for `ACCESS_KEY` and `CONTAINER_NAME`

## Data Structure in Azure

The workflow will upload data with this structure:
```
[container_name]/
  └── NVDA/
      └── NVDA_15min_pattern_ready_20250121.csv
      └── NVDA_15min_pattern_ready_20250128.csv
      └── ...
```

## Testing

After adding the secrets, you can manually trigger the `Update NVDA Data` workflow to test the Azure upload.

## Note

If the Azure secrets are not set, the workflow will still run successfully but will only save data locally in the GitHub Actions runner (which is temporary).