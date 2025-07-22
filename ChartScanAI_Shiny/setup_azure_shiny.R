#!/usr/bin/env Rscript

# Setup script for Azure-connected Shiny dashboard
# Installs required packages

cat("Installing packages for Azure-connected Shiny dashboard...\n\n")

# CRAN packages
packages <- c(
  "shiny",
  "shinydashboard",
  "plotly",
  "DT",
  "jsonlite",
  "httr",
  "lubridate",
  "dplyr",
  "AzureStor"
)

# Install packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  } else {
    cat(sprintf("✓ %s already installed\n", pkg))
  }
}

cat("\n✅ All packages installed!\n")
cat("\nTo run the Azure-connected dashboard:\n")
cat("  1. Add your Azure credentials to config/.env\n")
cat("  2. Run: shiny::runApp('btc_monitor_azure_advanced.R')\n")