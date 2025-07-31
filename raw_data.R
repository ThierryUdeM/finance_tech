library(reticulate)
library(dplyr)
library(arrow)
library(readr)
library(lubridate)
library(AzureStor)
library(TTR)
library(zoo)

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

# Ensure Python is configured
use_python("/usr/bin/python3", required = TRUE)

# Check if yfinance is installed, if not install it
if (!py_module_available("yfinance")) {
  system("pip3 install yfinance")
}

# Import the module
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
    
    # Define blob names
    current_blob_name <- paste0("raw_data/raw_data_", interval_name, ".parquet")
    historic_blob_name <- paste0("raw_data/historic_raw_data_", interval_name, ".parquet")
    
    tryCatch({
      # Upload current data (overwrites existing)
      storage_upload(container, temp_file, current_blob_name)
      cat("Uploaded current", interval_name, "data to:", current_blob_name, "\n")
      
      # Handle historic data
      # First, try to download existing historic data
      historic_exists <- tryCatch({
        temp_historic <- tempfile(fileext = ".parquet")
        storage_download(container, src = historic_blob_name, dest = temp_historic)
        TRUE
      }, error = function(e) {
        FALSE
      })
      
      if (historic_exists) {
        # Read existing historic data
        existing_historic <- read_parquet(temp_historic)
        
        # Remove any existing data for today
        today_date <- as.Date(last_trading_day)
        existing_historic_clean <- existing_historic[as.Date(existing_historic$datetime) != today_date, ]
        
        # Combine with today's new data
        updated_historic <- rbind(existing_historic_clean, combined_data)
        
        # Write and upload updated historic data
        temp_updated_historic <- tempfile(fileext = ".parquet")
        write_parquet(updated_historic, temp_updated_historic)
        storage_upload(container, temp_updated_historic, historic_blob_name)
        cat("Updated historic", interval_name, "data with", nrow(combined_data), "new records\n")
        
        # Clean up temp files
        unlink(temp_historic)
        unlink(temp_updated_historic)
      } else {
        # No existing historic data, create new
        storage_upload(container, temp_file, historic_blob_name)
        cat("Created new historic", interval_name, "data file with", nrow(combined_data), "records\n")
      }
      
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

# Calculate intraday indicators
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("Calculating intraday indicators...\n")

# Function to calculate intraday indicators
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
      true_range = high - low
    ) %>%
    ungroup()
  
  # ATR calculation per ticker
  for (ticker_name in unique(data$ticker)) {
    ticker_mask <- data$ticker == ticker_name
    ticker_data <- data[ticker_mask, ]
    
    if (nrow(ticker_data) >= 14) {
      atr_result <- ATR(cbind(ticker_data$high, ticker_data$low, ticker_data$close), n = 14)
      data$atr_14[ticker_mask] <- atr_result[, "atr"]
    } else {
      data$atr_14[ticker_mask] <- NA
    }
  }
  
  # NR4 / NR7
  data <- data %>%
    mutate(range = high - low) %>%
    group_by(ticker) %>%
    mutate(
      nr4 = if(n() >= 4) rollapply(range, width = 4, FUN = function(x) x[4] == min(x), align = "right", fill = NA) else NA,
      nr7 = if(n() >= 7) rollapply(range, width = 7, FUN = function(x) x[7] == min(x), align = "right", fill = NA) else NA
    ) %>%
    ungroup()
  
  # Bollinger Bands per ticker
  for (ticker_name in unique(data$ticker)) {
    ticker_mask <- data$ticker == ticker_name
    ticker_data <- data[ticker_mask, ]
    
    if (nrow(ticker_data) >= 20) {
      bb <- BBands(ticker_data$close, n = 20, sd = 2)
      data$bb_upper[ticker_mask] <- bb[, "up"]
      data$bb_middle[ticker_mask] <- bb[, "mavg"]
      data$bb_lower[ticker_mask] <- bb[, "dn"]
      data$bb_pctb[ticker_mask] <- bb[, "pctB"]
    } else {
      data$bb_upper[ticker_mask] <- NA
      data$bb_middle[ticker_mask] <- NA
      data$bb_lower[ticker_mask] <- NA
      data$bb_pctb[ticker_mask] <- NA
    }
  }
  
  data$bb_pierce <- (data$close > data$bb_upper) | (data$close < data$bb_lower)
  data$bb_pierce[is.na(data$bb_upper)] <- FALSE
  
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

# Function to save indicators data to Azure
save_indicators_to_azure <- function(data, interval_name) {
  if (nrow(data) > 0) {
    # Create temporary file
    temp_file <- tempfile(fileext = ".parquet")
    write_parquet(data, temp_file)
    
    # Define blob names in indicators_azure folder
    current_blob_name <- paste0("indicators_azure/data_feed_", interval_name, ".parquet")
    historic_blob_name <- paste0("indicators_azure/historic_data_feed_", interval_name, ".parquet")
    
    tryCatch({
      # Upload current data with indicators
      storage_upload(container, temp_file, current_blob_name)
      cat("Uploaded indicators data to:", current_blob_name, "\n")
      
      # Handle historic data with indicators
      # First, try to download existing historic indicators data
      historic_exists <- tryCatch({
        temp_historic <- tempfile(fileext = ".parquet")
        storage_download(container, src = historic_blob_name, dest = temp_historic)
        TRUE
      }, error = function(e) {
        FALSE
      })
      
      if (historic_exists) {
        # Read existing historic indicators data
        existing_historic <- read_parquet(temp_historic)
        
        # Remove any existing data for today
        today_date <- as.Date(last_trading_day)
        existing_historic_clean <- existing_historic[as.Date(existing_historic$datetime) != today_date, ]
        
        # Combine with today's new indicators data
        updated_historic <- rbind(existing_historic_clean, data)
        
        # Write and upload updated historic indicators data
        temp_updated_historic <- tempfile(fileext = ".parquet")
        write_parquet(updated_historic, temp_updated_historic)
        storage_upload(container, temp_updated_historic, historic_blob_name)
        cat("Updated historic indicators data with", nrow(data), "new records\n")
        
        # Clean up temp files
        unlink(temp_historic)
        unlink(temp_updated_historic)
      } else {
        # No existing historic indicators data, create new
        storage_upload(container, temp_file, historic_blob_name)
        cat("Created new historic indicators data file with", nrow(data), "records\n")
      }
      
      cat("Total records with indicators:", nrow(data), "\n")
      
    }, error = function(e) {
      cat("Error uploading indicators to Azure:", e$message, "\n")
    }, finally = {
      # Clean up temp file
      unlink(temp_file)
    })
    
  } else {
    cat("No indicators data to save for", interval_name, "\n")
  }
}

# Calculate and save 1-minute indicators
if (nrow(data_1min) > 0) {
  cat("\nCalculating 1-minute indicators...\n")
  data_1min_indicators <- calculate_intraday_indicators(data_1min)
  save_indicators_to_azure(data_1min_indicators, "1min")
}

# Calculate and save 5-minute indicators  
if (nrow(data_5min) > 0) {
  cat("\nCalculating 5-minute indicators...\n")
  data_5min_indicators <- calculate_intraday_indicators(data_5min)
  save_indicators_to_azure(data_5min_indicators, "5min")
}

cat("\nIntraday indicators calculation completed.\n")