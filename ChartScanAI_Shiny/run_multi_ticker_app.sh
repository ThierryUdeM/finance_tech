#!/bin/bash

# Script to run the Multi-Ticker Monitor Shiny App

echo "Starting Multi-Ticker Monitor Shiny App..."
echo "================================="

# Check if R is installed
if ! command -v R &> /dev/null; then
    echo "Error: R is not installed. Please install R first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "multi_ticker_monitor_azure.R" ]; then
    echo "Error: multi_ticker_monitor_azure.R not found in current directory"
    echo "Please run this script from the ChartScanAI_Shiny directory"
    exit 1
fi

# Check if config exists
if [ ! -f "config/.env" ]; then
    echo "Warning: config/.env not found"
    echo "Please create it with your Azure credentials:"
    echo "  STORAGE_ACCOUNT_NAME=your_account"
    echo "  ACCESS_KEY=your_key"
    echo "  CONTAINER_NAME=your_container"
    echo ""
fi

# Install required packages if needed
echo "Checking R packages..."
R --slave -e "
packages <- c('shiny', 'shinydashboard', 'plotly', 'DT', 'jsonlite', 
              'AzureStor', 'lubridate', 'dplyr', 'tidyr')
new_packages <- packages[!(packages %in% installed.packages()[,'Package'])]
if(length(new_packages)) {
  cat('Installing missing packages:', new_packages, '\n')
  install.packages(new_packages, repos='https://cloud.r-project.org/')
} else {
  cat('All required packages are installed\n')
}
"

# Run the app
echo ""
echo "Launching Multi-Ticker Monitor..."
echo "The app will open in your web browser"
echo "Press Ctrl+C to stop the app"
echo ""

R -e "shiny::runApp('multi_ticker_monitor_azure.R', launch.browser = TRUE)"