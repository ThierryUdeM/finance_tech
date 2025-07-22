# Diagnostic script to test Azure connection and data retrieval for multi-ticker app

library(AzureStor)
library(jsonlite)
library(lubridate)

cat("=== Multi-Ticker Azure Connection Test ===\n\n")

# Step 1: Load credentials
cat("1. Loading credentials...\n")
if (file.exists("config/.env")) {
  readRenviron("config/.env")
  cat("   ✓ Found config/.env\n")
} else {
  cat("   ✗ config/.env not found!\n")
  cat("   Please create config/.env with:\n")
  cat("   STORAGE_ACCOUNT_NAME=your_account\n")
  cat("   ACCESS_KEY=your_key\n")
  cat("   CONTAINER_NAME=your_container\n")
  stop("Missing configuration file")
}

# Check environment variables
storage_account <- Sys.getenv("STORAGE_ACCOUNT_NAME")
access_key <- Sys.getenv("ACCESS_KEY")
container_name <- Sys.getenv("CONTAINER_NAME")

cat("\n2. Checking environment variables...\n")
cat(sprintf("   Storage Account: %s\n", ifelse(nchar(storage_account) > 0, "✓ Set", "✗ Missing")))
cat(sprintf("   Access Key: %s\n", ifelse(nchar(access_key) > 0, "✓ Set", "✗ Missing")))
cat(sprintf("   Container Name: %s\n", ifelse(nchar(container_name) > 0, container_name, "✗ Missing")))

if (nchar(storage_account) == 0 || nchar(access_key) == 0 || nchar(container_name) == 0) {
  stop("Missing required environment variables")
}

# Step 2: Test Azure connection
cat("\n3. Testing Azure connection...\n")
tryCatch({
  blob_endpoint <- blob_endpoint(
    endpoint = sprintf("https://%s.blob.core.windows.net", storage_account),
    key = access_key
  )
  cat("   ✓ Created blob endpoint\n")
  
  container <- storage_container(blob_endpoint, container_name)
  cat("   ✓ Connected to container:", container_name, "\n")
}, error = function(e) {
  cat("   ✗ Connection failed:", e$message, "\n")
  stop("Could not connect to Azure")
})

# Step 3: List available data
cat("\n4. Checking available data in container...\n")
tickers <- c("BTC-USD", "NVDA", "AC.TO")

for (ticker in tickers) {
  cat(sprintf("\n   Checking %s:\n", ticker))
  
  # Try to list prediction files
  prefix <- sprintf("predictions/%s/", ticker)
  tryCatch({
    blobs <- list_blobs(container, prefix = prefix)
    if (!is.null(blobs) && nrow(blobs) > 0) {
      cat(sprintf("   ✓ Found %d prediction files\n", nrow(blobs)))
      # Show latest 3 files
      recent_blobs <- tail(blobs[order(blobs$name),], 3)
      for (i in 1:nrow(recent_blobs)) {
        cat(sprintf("     - %s\n", recent_blobs$name[i]))
      }
    } else {
      cat("   ✗ No prediction files found\n")
    }
  }, error = function(e) {
    cat("   ✗ Error listing blobs:", e$message, "\n")
  })
}

# Step 4: Try to download latest predictions
cat("\n5. Testing data download for each ticker...\n")
current_time <- with_tz(Sys.time(), "UTC")

for (ticker in tickers) {
  cat(sprintf("\n   Testing %s:\n", ticker))
  
  # Try current hour and previous hours
  found <- FALSE
  for (h_offset in 0:5) {
    check_time <- current_time - hours(h_offset)
    blob_path <- sprintf("predictions/%s/%s/%s.json", 
                        ticker,
                        format(check_time, "%Y-%m-%d"),
                        format(check_time, "%H"))
    
    cat(sprintf("   Trying: %s\n", blob_path))
    
    tryCatch({
      temp_file <- tempfile()
      storage_download(container, blob_path, temp_file)
      data <- fromJSON(temp_file)
      unlink(temp_file)
      
      cat("   ✓ SUCCESS! Found data:\n")
      cat(sprintf("     - Price: $%.2f\n", data$price))
      cat(sprintf("     - Recommendation: %s\n", data$recommendation))
      cat(sprintf("     - Timestamp: %s\n", data$timestamp))
      found <- TRUE
      break
    }, error = function(e) {
      # Continue to next hour
    })
  }
  
  if (!found) {
    cat("   ✗ No recent predictions found in last 6 hours\n")
  }
}

# Step 5: Check evaluation data
cat("\n6. Checking evaluation data...\n")
for (ticker in tickers) {
  prefix <- sprintf("evaluations/%s/", ticker)
  tryCatch({
    blobs <- list_blobs(container, prefix = prefix, max_results = 5)
    if (!is.null(blobs) && nrow(blobs) > 0) {
      cat(sprintf("   ✓ %s: Found %d evaluation files\n", ticker, nrow(blobs)))
    } else {
      cat(sprintf("   ✗ %s: No evaluation files found\n", ticker))
    }
  }, error = function(e) {
    cat(sprintf("   ✗ %s: Error - %s\n", ticker, e$message))
  })
}

# Step 6: Test alternative blob paths (in case of different structure)
cat("\n7. Checking alternative data structures...\n")

# Check root level
tryCatch({
  root_blobs <- list_blobs(container, max_results = 20)
  if (!is.null(root_blobs) && nrow(root_blobs) > 0) {
    cat("   Found these folders/files at root:\n")
    unique_dirs <- unique(dirname(root_blobs$name))
    for (dir in head(unique_dirs, 10)) {
      cat(sprintf("     - %s\n", dir))
    }
  }
}, error = function(e) {
  cat("   Error listing root:", e$message, "\n")
})

cat("\n=== Diagnostic Summary ===\n")
cat("If data is not loading in the app, check:\n")
cat("1. Are predictions being generated? (run multi_ticker_predictor_azure.py)\n")
cat("2. Is the data structure correct? (predictions/{ticker}/YYYY-MM-DD/HH.json)\n")
cat("3. Are the timestamps in UTC?\n")
cat("4. Do you have the correct container name?\n")

# Optional: Show current UTC time for reference
cat(sprintf("\nCurrent UTC time: %s\n", format(current_time, "%Y-%m-%d %H:%M:%S")))