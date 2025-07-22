#!/usr/bin/env Rscript
# Check what data is available in Azure

library(AzureStor)

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

# Check for predictions
cat("Checking for predictions...\n")
pred_blobs <- list_blobs(container, prefix = "predictions/")
if (!is.null(pred_blobs) && nrow(pred_blobs) > 0) {
  cat("Found", nrow(pred_blobs), "prediction files\n")
} else {
  cat("No predictions found\n")
}

# Check for evaluations
cat("\nChecking for evaluations...\n")
eval_blobs <- list_blobs(container, prefix = "evaluations/")
if (!is.null(eval_blobs) && nrow(eval_blobs) > 0) {
  cat("Found", nrow(eval_blobs), "evaluation files\n")
} else {
  cat("No evaluations found\n")
}

# Check for reports
cat("\nChecking for reports...\n")
report_blobs <- list_blobs(container, prefix = "reports/")
if (!is.null(report_blobs) && nrow(report_blobs) > 0) {
  cat("Found", nrow(report_blobs), "report files\n")
} else {
  cat("No reports found\n")
}

# Show all top-level directories
cat("\nAll top-level items in container:\n")
all_blobs <- list_blobs(container)
if (!is.null(all_blobs) && nrow(all_blobs) > 0) {
  # Extract top-level directories
  paths <- all_blobs$name
  top_level <- unique(sapply(strsplit(paths, "/"), function(x) x[1]))
  for (item in top_level) {
    cat(" -", item, "\n")
  }
}