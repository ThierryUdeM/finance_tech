# Fix GitHub Actions Permissions Error

## The Issue
Your workflows are getting "Permission denied" errors when trying to push commits back to the repository.

## Solution 1: Repository Settings (Try This First)

1. Go to: https://github.com/ThierryUdeM/finance_tech/settings/actions
2. Under "Workflow permissions", select **"Read and write permissions"**
3. Click "Save"

## Solution 2: Use Personal Access Token (If Solution 1 doesn't work)

### Step 1: Create a Personal Access Token
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name like "GitHub Actions"
4. Select these permissions:
   - `repo` (all)
   - `workflow`
5. Click "Generate token"
6. Copy the token (you won't see it again!)

### Step 2: Add Token to Repository Secrets
1. Go to: https://github.com/ThierryUdeM/finance_tech/settings/secrets/actions
2. Click "New repository secret"
3. Name: `PAT_TOKEN`
4. Value: [paste your token]
5. Click "Add secret"

### Step 3: Update Workflow
Replace the checkout step in your workflows with:
```yaml
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.PAT_TOKEN }}
```

## Solution 3: Remove Commit Steps (Quick Fix)
If you don't need to commit results back to the repo, you can comment out or remove the commit steps in the workflows.