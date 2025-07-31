#!/usr/bin/env Rscript

# Calculate indicators for intraday data (1min and 5min)
# This script pulls raw data from Azure and calculates technical indicators

library(TTR)
library(dplyr)
library(zoo)
library(arrow)
library(AzureStor)
library(lubridate)

# Get Azure credentials from environment
azure_account <- Sys.getenv("STORAGE_ACCOUNT_NAME")
azure_container <- Sys.getenv("CONTAINER_NAME")
azure_key <- Sys.getenv("ACCESS_KEY")

if (azure_account == "" || azure_container == "" || azure_key == "") {
  stop("Azure credentials not found in environment variables.")
}

# Connect to Azure
endpoint <- storage_endpoint(
  paste0("https://", azure_account, ".blob.core.windows.net"),
  key = azure_key
)
container <- storage_container(endpoint, azure_container)

# Function to calculate intraday indicators (same as original indicators.R)
calculate_intraday_indicators <- function(data) {
  # Ensure data is sorted by datetime
  data <- data[order(data$ticker, data$datetime), ]
  
  if (!inherits(data$datetime, "POSIXct")) {
    data$datetime <- as.POSIXct(data$datetime)
  }
  
  # Add date column for grouping
  data$date <- as.Date(data$datetime)
  
  # Price levels & ranges (session-specific, reset daily)
  data <- data %>%
    group_by(ticker, date) %>%
    mutate(opening_price = first(open)) %>%
    ungroup()
  
  # Prior close
  data <- data %>%
    group_by(ticker) %>%
    mutate(prior_close = lag(close, 1)) %>%
    ungroup()
  
  # Running HOD/LOD with daily reset
  data <- data %>%
    group_by(ticker, date) %>%
    mutate(
      running_hod = cummax(high),
      running_lod = cummin(low)
    ) %>%
    ungroup()
  
  # Opening Range High & Low (first 3 bars for 15-minute ORB on 5-min data)
  data <- data %>%
    group_by(ticker, date) %>%
    mutate(
      or_high = max(high[1:min(3, n())]),
      or_low = min(low[1:min(3, n())])
    ) %>%
    ungroup()
  
  # Trend Moving Averages
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      ema_9 = if(n() >= 9) EMA(close, n = 9) else NA,
      ema_20 = if(n() >= 20) EMA(close, n = 20) else NA,
      sma_5 = if(n() >= 5) SMA(close, n = 5) else NA,
      sma_20 = if(n() >= 20) SMA(close, n = 20) else NA
    ) %>%
    ungroup()
  
  # Momentum
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      rsi_14 = if(n() >= 14) RSI(close, n = 14) else NA,
      rsi_slope = c(NA, diff(rsi_14))
    ) %>%
    ungroup()
  
  # Volatility / Range
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      true_range = high - low,
      atr_14 = if(n() >= 14) ATR(cbind(high, low, close), n = 14)[, "atr"] else NA
    ) %>%
    ungroup()
  
  # NR4 / NR7
  data <- data %>%
    mutate(range = high - low) %>%
    group_by(ticker) %>%
    mutate(
      nr4 = if(n() >= 4) rollapply(range, width = 4, FUN = function(x) x[4] == min(x), align = "right", fill = NA) else NA,
      nr7 = if(n() >= 7) rollapply(range, width = 7, FUN = function(x) x[7] == min(x), align = "right", fill = NA) else NA
    ) %>%
    ungroup()
  
  # Bollinger Bands
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      bb_result = if(n() >= 20) list(BBands(close, n = 20, sd = 2)) else list(NULL)
    ) %>%
    ungroup()
  
  # Extract BB components
  for (i in 1:nrow(data)) {
    if (!is.null(data$bb_result[[i]])) {
      data$bb_upper[i] <- data$bb_result[[i]][i, "up"]
      data$bb_middle[i] <- data$bb_result[[i]][i, "mavg"]
      data$bb_lower[i] <- data$bb_result[[i]][i, "dn"]
      data$bb_pctb[i] <- data$bb_result[[i]][i, "pctB"]
    } else {
      data$bb_upper[i] <- NA
      data$bb_middle[i] <- NA
      data$bb_lower[i] <- NA
      data$bb_pctb[i] <- NA
    }
  }
  data$bb_result <- NULL
  data$bb_pierce <- (data$close > data$bb_upper) | (data$close < data$bb_lower)
  
  # Volume metrics
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      avg_volume_20 = if(n() >= 20) SMA(volume, n = 20) else NA,
      volume_spike_ratio = volume / avg_volume_20,
      obv = OBV(close, volume)
    ) %>%
    ungroup()
  
  # Intraday VWAP (reset daily)
  data <- data %>%
    group_by(ticker, date) %>%
    mutate(
      typical_price = (high + low + close) / 3,
      vwap = cumsum(typical_price * volume) / cumsum(volume)
    ) %>%
    ungroup() %>%
    select(-typical_price)
  
  # Gap metric
  data$gap_pct <- ifelse(!is.na(data$prior_close), 
                         (data$open - data$prior_close) / data$prior_close * 100, 
                         NA)
  
  # Price-action flags
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      inside_bar = high <= lag(high) & low >= lag(low)
    ) %>%
    ungroup()
  
  # Hammer detection
  body <- abs(data$close - data$open)
  lower_shadow <- abs(ifelse(data$close >= data$open, data$open - data$low, data$close - data$low))
  upper_shadow <- abs(ifelse(data$close >= data$open, data$high - data$close, data$high - data$open))
  data$hammer <- (lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)
  
  # Engulfing pattern
  data <- data %>%
    group_by(ticker) %>%
    mutate(
      bull_engulf = (close > open) & 
                    (lag(close) < lag(open)) & 
                    (close > lag(open)) & 
                    (open < lag(close)),
      bear_engulf = (close < open) & 
                    (lag(close) > lag(open)) & 
                    (close < lag(open)) & 
                    (open > lag(close)),
      engulfing = bull_engulf | bear_engulf
    ) %>%
    select(-bull_engulf, -bear_engulf) %>%
    ungroup()
  
  return(data)
}

# Function to load data from Azure
load_from_azure <- function(blob_name) {
  tryCatch({
    temp_file <- tempfile(fileext = ".parquet")
    storage_download(container, src = blob_name, dest = temp_file)
    data <- read_parquet(temp_file)
    unlink(temp_file)
    return(data)
  }, error = function(e) {
    cat("Error loading", blob_name, ":", e$message, "\n")
    return(NULL)
  })
}

# Function to save data to Azure
save_to_azure <- function(data, blob_name) {
  tryCatch({
    temp_file <- tempfile(fileext = ".parquet")
    write_parquet(data, temp_file)
    storage_upload(container, temp_file, blob_name)
    unlink(temp_file)
    cat("Saved", nrow(data), "records to", blob_name, "\n")
    return(TRUE)
  }, error = function(e) {
    cat("Error saving to", blob_name, ":", e$message, "\n")
    return(FALSE)
  })
}

# Main execution
cat("Starting intraday indicator calculation...\n")

# Process 1-minute data
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Processing 1-minute data...\n")

raw_1min <- load_from_azure("raw_data/raw_data_1min.parquet")
if (!is.null(raw_1min)) {
  data_1min <- calculate_intraday_indicators(raw_1min)
  save_to_azure(data_1min, "processed_data/data_feed_1min.parquet")
  
  # Also process historic data
  historic_1min <- load_from_azure("raw_data/historic_raw_data_1min.parquet")
  if (!is.null(historic_1min)) {
    historic_data_1min <- calculate_intraday_indicators(historic_1min)
    save_to_azure(historic_data_1min, "processed_data/historic_data_feed_1min.parquet")
  }
}

# Process 5-minute data
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Processing 5-minute data...\n")

raw_5min <- load_from_azure("raw_data/raw_data_5min.parquet")
if (!is.null(raw_5min)) {
  data_5min <- calculate_intraday_indicators(raw_5min)
  save_to_azure(data_5min, "processed_data/data_feed_5min.parquet")
  
  # Also process historic data
  historic_5min <- load_from_azure("raw_data/historic_raw_data_5min.parquet")
  if (!is.null(historic_5min)) {
    historic_data_5min <- calculate_intraday_indicators(historic_5min)
    save_to_azure(historic_data_5min, "processed_data/historic_data_feed_5min.parquet")
  }
}

cat("\nIntraday indicator calculation completed.\n")