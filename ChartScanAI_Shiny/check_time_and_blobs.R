#!/usr/bin/env Rscript
# Check current time and available prediction blobs

library(AzureStor)
library(lubridate)

# Load environment variables
if (file.exists("../config/.env")) {
  readRenviron("../config/.env")
} else if (file.exists("config/.env")) {
  readRenviron("config/.env")
}

# Connect to Azure
blob_endpoint <- blob_endpoint(
  endpoint = sprintf("https://%s.blob.core.windows.net", 
                    Sys.getenv("AZURE_STORAGE_ACCOUNT")),
  key = Sys.getenv("AZURE_STORAGE_KEY")
)

container <- storage_container(blob_endpoint, Sys.getenv("AZURE_CONTAINER_NAME"))

# Show current time info
cat("Current system time:", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"), "\n")
cat("Current UTC time:", format(with_tz(Sys.time(), "UTC"), "%Y-%m-%d %H:%M:%S"), "\n\n")

# List all prediction blobs
cat("Available prediction blobs:\n")
pred_blobs <- list_blobs(container, prefix = "predictions/")
if (!is.null(pred_blobs) && nrow(pred_blobs) > 0) {
  for(i in 1:nrow(pred_blobs)) {
    cat(" -", pred_blobs$name[i], "\n")
  }
  
  # Try to read the most recent one
  cat("\nTrying to read the most recent prediction...\n")
  latest_blob <- pred_blobs$name[nrow(pred_blobs)]
  
  tryCatch({
    temp_file <- tempfile()
    storage_download(container, latest_blob, temp_file)
    data <- jsonlite::fromJSON(temp_file)
    unlink(temp_file)
    
    cat("\nPrediction content:\n")
    cat("Timestamp:", data$timestamp, "\n")
    cat("Price:", data$price, "\n")
    cat("Recommendation:", data$recommendation, "\n")
    cat("Total buy signals:", data$total_buy_signals, "\n")
    cat("Total sell signals:", data$total_sell_signals, "\n")
  }, error = function(e) {
    cat("Error reading prediction:", e$message, "\n")
  })
}