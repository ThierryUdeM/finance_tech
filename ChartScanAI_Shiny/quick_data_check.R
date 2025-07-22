# Quick script to check if multi-ticker data exists in Azure

library(AzureStor)
library(jsonlite)
library(lubridate)

# Load config
readRenviron("config/.env")

# Get credentials (handle both naming conventions)
storage_account <- Sys.getenv("AZURE_STORAGE_ACCOUNT", Sys.getenv("STORAGE_ACCOUNT_NAME"))
access_key <- Sys.getenv("AZURE_STORAGE_KEY", Sys.getenv("ACCESS_KEY"))
container_name <- Sys.getenv("AZURE_CONTAINER_NAME", Sys.getenv("CONTAINER_NAME"))

cat("Connecting to Azure...\n")
cat("Account:", storage_account, "\n")
cat("Container:", container_name, "\n\n")

# Connect
blob_endpoint <- blob_endpoint(
  endpoint = sprintf("https://%s.blob.core.windows.net", storage_account),
  key = access_key
)
container <- storage_container(blob_endpoint, container_name)

# Check for data
tickers <- c("BTC-USD", "NVDA", "AC.TO")
current_time <- with_tz(Sys.time(), "UTC")

cat("Current UTC time:", format(current_time, "%Y-%m-%d %H:%M:%S"), "\n\n")

for (ticker in tickers) {
  cat("=== Checking", ticker, "===\n")
  
  # Look for recent predictions
  found <- FALSE
  for (h in 0:24) {
    check_time <- current_time - hours(h)
    blob_path <- sprintf("predictions/%s/%s/%s.json", 
                        ticker,
                        format(check_time, "%Y-%m-%d"),
                        format(check_time, "%H"))
    
    tryCatch({
      temp_file <- tempfile()
      storage_download(container, blob_path, temp_file, overwrite = TRUE)
      data <- fromJSON(temp_file)
      unlink(temp_file)
      
      cat("✓ Found data from", h, "hours ago:\n")
      cat("  Path:", blob_path, "\n")
      cat("  Price: $", data$price, "\n")
      cat("  Recommendation:", data$recommendation, "\n")
      cat("  Buy signals:", data$total_buy_signals, "\n")
      cat("  Sell signals:", data$total_sell_signals, "\n\n")
      found <- TRUE
      break
    }, error = function(e) {
      # Continue searching
    })
  }
  
  if (!found) {
    cat("✗ No data found in last 24 hours\n\n")
    
    # List what's actually there
    cat("Checking what exists for", ticker, ":\n")
    prefix <- sprintf("predictions/%s/", ticker)
    tryCatch({
      blobs <- list_blobs(container, prefix = prefix, max_results = 10)
      if (!is.null(blobs) && nrow(blobs) > 0) {
        cat("Found these files:\n")
        for (i in 1:min(5, nrow(blobs))) {
          cat(" -", blobs$name[i], "\n")
        }
      } else {
        cat("No files found with prefix:", prefix, "\n")
      }
    }, error = function(e) {
      cat("Error listing blobs:", e$message, "\n")
    })
    cat("\n")
  }
}

cat("\nIf no data found, you may need to:\n")
cat("1. Run the multi_ticker_predictor_azure.py script first\n")
cat("2. Check if data is being saved to the correct container\n")
cat("3. Verify the ticker names match (BTC-USD, NVDA, AC.TO)\n")