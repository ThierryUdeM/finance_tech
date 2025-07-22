#!/bin/bash

# Run Bitcoin Monitor Shiny App

echo "Starting Bitcoin Monitor Dashboard..."
echo ""
echo "The dashboard will be available at: http://localhost:5001"
echo "Press Ctrl+C to stop the server."
echo ""

# Run the Shiny app on a different port than the main app
R -e "shiny::runApp('btc_monitor_app.R', port = 5001, host = '0.0.0.0')"