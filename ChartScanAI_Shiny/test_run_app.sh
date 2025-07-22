#!/bin/bash
# Test running the Azure advanced Shiny app

echo "Starting BTC Monitor Azure Advanced app..."
echo "The app should open in your browser at http://localhost:3838"
echo "Press Ctrl+C to stop the app"
echo ""

# Change to the script directory
cd "$(dirname "$0")"

# Run the app
Rscript -e "shiny::runApp('btc_monitor_azure_advanced.R', host='0.0.0.0', port=3838)"