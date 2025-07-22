#!/bin/bash

# Run ChartScanAI Shiny App

echo "Starting ChartScanAI Shiny App..."
echo ""
echo "Make sure you have run setup.R first to install required packages."
echo ""
echo "The app will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server."
echo ""

# Run the Shiny app
R -e "shiny::runApp('app.R', port = 5000, host = '0.0.0.0')"