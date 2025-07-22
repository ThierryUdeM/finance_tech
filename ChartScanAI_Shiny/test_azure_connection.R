#!/usr/bin/env Rscript
# Test Azure connection for the Shiny app

library(AzureStor)

# Load environment variables
if (file.exists("../config/.env")) {
  readRenviron("../config/.env")
} else if (file.exists("config/.env")) {
  readRenviron("config/.env")
}

# Display loaded environment variables (hiding the key)
cat("Environment Variables:\n")
cat("AZURE_STORAGE_ACCOUNT:", Sys.getenv("AZURE_STORAGE_ACCOUNT"), "\n")
cat("AZURE_CONTAINER_NAME:", Sys.getenv("AZURE_CONTAINER_NAME"), "\n")
cat("AZURE_STORAGE_KEY:", if(nchar(Sys.getenv("AZURE_STORAGE_KEY")) > 0) "***" else "NOT FOUND", "\n\n")

# Test connection
tryCatch({
  cat("Creating blob endpoint...\n")
  blob_endpoint <- blob_endpoint(
    endpoint = sprintf("https://%s.blob.core.windows.net", 
                      Sys.getenv("AZURE_STORAGE_ACCOUNT")),
    key = Sys.getenv("AZURE_STORAGE_KEY")
  )
  
  cat("Getting container...\n")
  container <- storage_container(blob_endpoint, Sys.getenv("AZURE_CONTAINER_NAME"))
  
  cat("Connection successful!\n\n")
  
  # Try to list some blobs
  cat("Listing blobs in container...\n")
  blobs <- list_blobs(container)
  
  if (!is.null(blobs) && nrow(blobs) > 0) {
    cat("Found", nrow(blobs), "blobs:\n")
    cat("Column names:", paste(names(blobs), collapse=", "), "\n\n")
    # Print first few blob names
    cat("First few blobs:\n")
    for(i in 1:min(5, nrow(blobs))) {
      cat(" -", blobs$name[i], "\n")
    }
  } else {
    cat("No blobs found in container.\n")
  }
  
}, error = function(e) {
  cat("Error:", e$message, "\n")
  cat("\nPossible issues:\n")
  cat("1. Check that the storage account name is correct\n")
  cat("2. Verify the access key is valid\n")
  cat("3. Ensure the container name exists\n")
  cat("4. Check network connectivity\n")
})