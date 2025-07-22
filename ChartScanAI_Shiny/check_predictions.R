#!/usr/bin/env Rscript
# Check prediction files in Azure

library(AzureStor)
library(jsonlite)

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

# List prediction files
pred_blobs <- list_blobs(container, prefix = "predictions/")
if (!is.null(pred_blobs) && nrow(pred_blobs) > 0) {
  cat("Prediction files:\n")
  for(i in 1:nrow(pred_blobs)) {
    cat(" -", pred_blobs$name[i], "\n")
  }
  
  # Try to download and show the latest prediction
  cat("\nTrying to download the latest prediction...\n")
  latest_blob <- pred_blobs$name[nrow(pred_blobs)]
  
  tryCatch({
    temp_file <- tempfile()
    storage_download(container, latest_blob, temp_file)
    data <- fromJSON(temp_file)
    unlink(temp_file)
    
    cat("\nLatest prediction content:\n")
    cat("Timestamp:", data$timestamp, "\n")
    cat("Price:", data$price, "\n")
    cat("Recommendation:", data$recommendation, "\n")
    cat("Buy signals:", data$total_buy_signals, "\n")
    cat("Sell signals:", data$total_sell_signals, "\n")
  }, error = function(e) {
    cat("Error reading prediction:", e$message, "\n")
  })
}