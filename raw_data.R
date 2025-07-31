library(reticulate)
library(dplyr)
library(arrow)
library(readr)
library(lubridate)
library(AzureStor)

# Get environment variables
tickers_string <- Sys.getenv("TICKERS")
if (tickers_string == "") {
  stop("TICKERS environment variable not found.")
}

azure_account <- Sys.getenv("STORAGE_ACCOUNT_NAME")
azure_container <- Sys.getenv("CONTAINER_NAME")
azure_key <- Sys.getenv("ACCESS_KEY")

if (azure_account == "" || azure_container == "" || azure_key == "") {
  stop("Azure credentials not found in environment variables.")
}

# Split tickers into vector and create space-separated string for yfinance
tickers <- unlist(strsplit(tickers_string, ","))
tickers <- trimws(tickers)
tickers_for_yf <- paste(tickers, collapse = " ")

# Import yfinance
yf <- import("yfinance")

# Connect to Azure container
endpoint <- storage_endpoint(
  paste0("https://", azure_account, ".blob.core.windows.net"),
  key = azure_key
)
container <- storage_container(endpoint, azure_container)

# Determine the last trading day
get_last_trading_day <- function() {
  today <- Sys.Date()
  current_time <- Sys.time()
  weekday <- weekdays(today)
  
  # Market opens at 9:30 AM EDT
  market_open <- as.POSIXct(paste(today, "09:30:00"), tz = "America/New_York")
  
  # If before market open on a weekday, use previous trading day
  if (weekday %in% c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")) {
    if (current_time < market_open) {
      if (weekday == "Monday") {
        return(today - 3)  # Previous Friday
      } else {
        return(today - 1)  # Previous weekday
      }
    } else {
      return(today)  # Current weekday after market open
    }
  } else if (weekday == "Saturday") {
    return(today - 1)  # Friday
  } else if (weekday == "Sunday") {
    return(today - 2)  # Friday
  }
}

last_trading_day <- get_last_trading_day()
cat("Fetching data for last trading day:", as.character(last_trading_day), "\n")

# Function to fetch and process data for a given interval
fetch_and_process_data <- function(interval_name, interval_code) {
  cat("Fetching", interval_name, "OHLCV data for", length(tickers), "tickers...\n")
  
  tryCatch({
    # Use bulk download - more efficient
    data <- yf$download(
      tickers = tickers_for_yf,
      start = as.character(last_trading_day),
      end = as.character(last_trading_day + 1),
      interval = interval_code,
      progress = FALSE,
      auto_adjust = TRUE
    )
    
    if (nrow(data) == 0) {
      stop("No data retrieved for any tickers")
    }
    
    # Debug: show column structure
    cat("Column names:", paste(names(data), collapse = ", "), "\n")
    cat("Data dimensions:", nrow(data), "x", ncol(data), "\n")
    
    # Convert multi-level columns to long format
    combined_data <- data.frame()
    
    # Convert pandas DataFrame with MultiIndex columns to R format
    metrics <- c("Open", "High", "Low", "Close", "Volume")
    col_names <- names(data)
    
    for (ticker in tickers) {
      tryCatch({
        # Extract data for this ticker using pandas-style column access
        ticker_data_list <- list()
        
        for (metric in metrics) {
          col_name <- paste0("('", metric, "', '", ticker, "')")
          if (col_name %in% col_names) {
            ticker_data_list[[tolower(metric)]] <- data[[col_name]]
          } else {
            cat("Missing column:", col_name, "for ticker:", ticker, "\n")
            ticker_data_list[[tolower(metric)]] <- rep(NA, nrow(data))
          }
        }
        
        # yfinance returns data already in EST/EDT timezone
        et_datetime <- as.POSIXct(row.names(data))
        et_datetime <- lubridate::with_tz(as.POSIXct(row.names(data), tz = "UTC"), "America/New_York")
        
        ticker_data <- data.frame(
          ticker = ticker,
          datetime = et_datetime,
          open = ticker_data_list[["open"]],
          high = ticker_data_list[["high"]],
          low = ticker_data_list[["low"]],
          close = ticker_data_list[["close"]],
          volume = ticker_data_list[["volume"]],
          stringsAsFactors = FALSE
        )
        
        # Remove rows with all NA values
        ticker_data <- ticker_data[!is.na(ticker_data$close), ]
        
        if (nrow(ticker_data) > 0) {
          combined_data <- rbind(combined_data, ticker_data)
          cat("Processed", nrow(ticker_data), "records for", ticker, "\n")
        } else {
          cat("No valid data for", ticker, "(all NA values)\n")
        }
        
      }, error = function(e) {
        cat("Error processing", ticker, ":", e$message, "\n")
      })
    }
    
    return(combined_data)
    
  }, error = function(e) {
    cat("Error fetching", interval_name, "data:", e$message, "\n")
    return(data.frame())
  })
}

# Function to save data to Azure blob storage
save_to_azure <- function(combined_data, interval_name) {
  if (nrow(combined_data) > 0) {
    # Create temporary file
    temp_file <- tempfile(fileext = ".parquet")
    write_parquet(combined_data, temp_file)
    
    # Generate blob name with timestamp
    current_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
    blob_name <- paste0("raw_data/raw_data_", interval_name, "_", current_timestamp, ".parquet")
    
    # Also save a "latest" version for easy access
    latest_blob_name <- paste0("raw_data/raw_data_", interval_name, "_latest.parquet")
    
    tryCatch({
      # Upload timestamped version
      storage_upload(container, temp_file, blob_name)
      cat("Uploaded", interval_name, "data to Azure:", blob_name, "\n")
      
      # Upload latest version (overwrites existing)
      storage_upload(container, temp_file, latest_blob_name)
      cat("Updated latest", interval_name, "data:", latest_blob_name, "\n")
      
      cat("Total records:", nrow(combined_data), "\n")
      cat("Date range:", min(combined_data$datetime), "to", max(combined_data$datetime), "\n")
      cat("Unique tickers:", length(unique(combined_data$ticker)), "\n")
      
    }, error = function(e) {
      cat("Error uploading to Azure:", e$message, "\n")
    }, finally = {
      # Clean up temp file
      unlink(temp_file)
    })
    
  } else {
    cat("No valid data was processed for", interval_name, "\n")
  }
}

# Main execution
cat("Starting data fetch for both 1-minute and 5-minute intervals...\n")

# Fetch 1-minute data
data_1min <- fetch_and_process_data("1-minute", "1m")
save_to_azure(data_1min, "1min")

cat("\n", paste(rep("=", 60), collapse = ""), "\n")

# Fetch 5-minute data  
data_5min <- fetch_and_process_data("5-minute", "5m")
save_to_azure(data_5min, "5min")

cat("Data fetching completed for both intervals.\n")