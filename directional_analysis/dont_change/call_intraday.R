# Intraday Pattern Analysis Script
# This script uses historical pattern matching to predict short-term stock movements

# Clear any existing VIRTUAL_ENV to avoid conflicts
Sys.unsetenv("VIRTUAL_ENV")

# Load Reticulate
library(reticulate)

# Import the Python script
source_python("intraday_shape_matcher.py")

# Function to analyze a single ticker
analyze_ticker <- function(ticker, show_details = TRUE) {
  cat(sprintf("\n=== %s Pattern Analysis ===\n", ticker))
  
  # Analysis 1: Short-term (5-minute bars)
  if (show_details) cat("1. SHORT-TERM (5-minute bars, 2-day history):\n")
  
  res_5m <- tryCatch({
    forecast_shape(
      ticker        = ticker,
      interval      = "5m",
      period        = "2d",
      query_length  = 12,        # 1 hour of data
      K             = 5
    )
  }, error = function(e) {
    if (show_details) cat("   Error with 5m data:", conditionMessage(e), "\n")
    list(`1h` = NA, `3h` = NA, `eod` = NA)
  })
  
  if (show_details && !is.na(res_5m$`1h`)) {
    cat(sprintf("   - Next 1 hour:  %+.2f%%\n", res_5m$`1h` * 100))
    cat(sprintf("   - Next 3 hours: %+.2f%%\n", res_5m$`3h` * 100))
    cat(sprintf("   - End of day:   %+.2f%%\n", res_5m$`eod` * 100))
    cat("\n")
  }
  
  # Analysis 2: Medium-term (1-hour bars)
  if (show_details) cat("2. MEDIUM-TERM (1-hour bars, 60-day history):\n")
  
  res_1h <- tryCatch({
    forecast_shape(
      ticker        = ticker,
      interval      = "1h",
      period        = "60d",
      query_length  = 5,         # 5 hours of data
      K             = 5
    )
  }, error = function(e) {
    if (show_details) cat("   Error with 1h data:", conditionMessage(e), "\n")
    list(`1h` = NA, `3h` = NA, `eod` = NA)
  })
  
  if (show_details && !is.na(res_1h$`1h`)) {
    cat(sprintf("   - Next 1 hour:  %+.2f%%\n", res_1h$`1h` * 100))
    cat(sprintf("   - Next 3 hours: %+.2f%%\n", res_1h$`3h` * 100))
    cat(sprintf("   - End of day:   %+.2f%%\n", res_1h$`eod` * 100))
  }
  
  # Return combined results
  return(list(
    ticker = ticker,
    short_term = res_5m,
    medium_term = res_1h
  ))
}

# Function to display summary for multiple tickers
display_summary <- function(results) {
  cat("\n=== SUMMARY TABLE ===\n")
  cat("Pattern-based predictions:\n\n")
  cat("Ticker | Timeframe    | 1-hour | 3-hour | EOD\n")
  cat("-------|--------------|--------|--------|--------\n")
  
  for (res in results) {
    # Short-term results
    if (!is.na(res$short_term$`1h`)) {
      cat(sprintf("%-6s | 5-min bars   | %+.2f%% | %+.2f%% | %+.2f%%\n", 
          res$ticker,
          res$short_term$`1h` * 100, 
          res$short_term$`3h` * 100, 
          res$short_term$`eod` * 100))
    }
    
    # Medium-term results
    if (!is.na(res$medium_term$`1h`)) {
      cat(sprintf("%-6s | 1-hour bars  | %+.2f%% | %+.2f%% | %+.2f%%\n", 
          res$ticker,
          res$medium_term$`1h` * 100, 
          ifelse(is.na(res$medium_term$`3h`), 0, res$medium_term$`3h` * 100), 
          res$medium_term$`eod` * 100))
    }
  }
  
  cat("\nNote: These predictions are based on historical pattern matching.\n")
  cat("Past patterns do not guarantee future results.\n")
}

# Main execution
if (!interactive()) {
  # If run from command line with arguments
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) == 0) {
    # Default: analyze AAPL
    results <- list(analyze_ticker("AAPL"))
    display_summary(results)
  } else {
    # Analyze specified tickers
    results <- lapply(args, function(ticker) {
      analyze_ticker(ticker, show_details = (length(args) == 1))
    })
    display_summary(results)
  }
} else {
  # Interactive mode - provide examples
  cat("Intraday Pattern Matcher loaded.\n")
  cat("\nUsage examples:\n")
  cat("  # Single ticker with details:\n")
  cat("  result <- analyze_ticker('AAPL')\n")
  cat("\n  # Multiple tickers:\n")
  cat("  tickers <- c('AAPL', 'MSFT', 'GOOGL', 'TSLA')\n")
  cat("  results <- lapply(tickers, function(t) analyze_ticker(t, FALSE))\n")
  cat("  display_summary(results)\n")
  cat("\n  # Direct function call:\n")
  cat("  forecast_shape('AAPL', interval='1h', period='60d', query_length=5, K=5)\n")
}
