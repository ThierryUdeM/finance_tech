# GitHub Actions Secrets Setup

To enable the automated signal detection workflow, you need to add the following secrets to your GitHub repository:

## Required Secrets

1. **GMAIL_USER**
   - Your Gmail address (e.g., `thierry.gc@gmail.com`)

2. **GMAIL_APP_PWD**
   - Your Gmail App Password (16-character password without spaces)
   - Example: `rdnoqmgczyekduin`

3. **ALERT_TO**
   - Email address to receive alerts (e.g., `thierry.gc@gmail.com`)

4. **TICKERS** (Optional)
   - Comma-separated list of tickers to monitor
   - Default: `NVDA,MSFT,TSLA,AAPL,GOOG,META,AMZN`
   - You can add your Canadian tickers: `NVDA,MSFT,TSLA,GOOG,NA.TO,AC.TO,CNQ.TO,SU.TO,MFC.TO,WCP.TO,CVE.TO,ENB.TO,IVN.TO,GWO.TO,BTE.TO,HND.TO,DSV.TO,TD.TO,AGI.TO,RY.TO,SLF.TO,DML.TO,TLO.TO,ABX.TO,BAM.TO`

## How to Add Secrets

1. Go to your GitHub repository
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret**
5. Add each secret with its name and value
6. Click **Add secret**

## Testing the Workflow

Once secrets are added, you can test the workflow:

1. Go to **Actions** tab in your repository
2. Select **Trading Signal Detector** workflow
3. Click **Run workflow**
4. Enable **test_mode** to run outside market hours
5. Click **Run workflow** button

## Important Notes

- The workflow runs every 15 minutes during market hours (Mon-Fri, 9:30 AM - 4:00 PM ET)
- Signals are only sent for LONG positions with ≥78% confidence
- Email alerts include entry, stop, target, and R:R ratio
- All signal results are saved as artifacts for 7 days

## Monitoring

- Check the **Actions** tab to see workflow runs
- Each run shows a summary of detected signals
- Download `signals.json` artifact for detailed results