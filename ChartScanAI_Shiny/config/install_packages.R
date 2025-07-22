# =============================================================================
# Package Installation Script
# =============================================================================
# Install required packages for the financial analysis application

# List of required packages
required_packages <- c(
  "shiny",          # Web application framework
  "shinydashboard", # Dashboard UI components
  "plotly",         # Interactive charts
  "DT",             # Data tables
  "quantmod",       # Financial data and technical analysis
  "tidyquant",      # Financial data integration
  "TTR",            # Technical trading rules
  "dplyr",          # Data manipulation
  "lubridate",      # Date/time handling
  "httr",           # HTTP client
  "jsonlite",       # JSON parsing
  "base64enc",      # Base64 encoding
  "reticulate",     # Python integration
  "ggplot2",        # Static plotting
  "webshot"         # Web screenshot capability
)

# Install packages if not already installed
for (package in required_packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, repos = "https://cran.r-project.org/")
    library(package, character.only = TRUE)
  }
}

# Install phantomjs for webshot (if needed)
if (!webshot::is_phantomjs_installed()) {
  webshot::install_phantomjs()
}

cat("All required packages installed successfully!\n")