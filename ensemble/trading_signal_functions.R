# Trading Signal Display Functions for R Shiny
# Helper functions to display actionable trading signals

library(jsonlite)
library(DT)
library(shiny)
library(dplyr)

# Function to load trading signals from JSON
load_trading_signals <- function(file_path = "trading_signals_latest.json") {
  if (file.exists(file_path)) {
    signals <- fromJSON(file_path, flatten = TRUE)
    return(signals)
  } else {
    return(NULL)
  }
}

# Function to load summary from JSON
load_signal_summary <- function(file_path = "actionable_summary_latest.json") {
  if (file.exists(file_path)) {
    summary <- fromJSON(file_path, flatten = TRUE)
    return(summary)
  } else {
    return(NULL)
  }
}

# Create signal card UI element
create_signal_card <- function(signal) {
  if (is.null(signal)) return(NULL)
  
  # Determine card color based on signal type
  card_color <- switch(signal$signal_type,
    "BUY" = "success",
    "SELL" = "danger",
    "primary"
  )
  
  # Determine confidence badge color
  conf_color <- if (signal$confidence >= 0.7) "success" 
                else if (signal$confidence >= 0.5) "warning" 
                else "secondary"
  
  # Create card HTML
  card_html <- div(
    class = paste("card border", card_color, "mb-3"),
    div(
      class = "card-header",
      h4(
        class = "card-title mb-0",
        span(signal$ticker, class = "font-weight-bold"),
        span(
          signal$signal_type,
          class = paste("badge badge", card_color, "float-right")
        )
      )
    ),
    div(
      class = "card-body",
      # Current price and VWAP
      div(
        class = "row mb-2",
        div(
          class = "col-6",
          strong("Current Price: "), 
          span(paste0("$", signal$current_price))
        ),
        div(
          class = "col-6",
          strong("VWAP: "), 
          span(paste0("$", signal$vwap))
        )
      ),
      
      # Entry bands (if signal exists)
      if (!is.null(signal$entry_bands)) {
        div(
          class = "alert alert-info",
          h5("Entry Zone"),
          div(
            class = "row",
            div(
              class = "col-6",
              strong("Entry Range: "),
              br(),
              span(
                paste0("$", signal$entry_bands$entry_low, " - $", signal$entry_bands$entry_high),
                class = "text-primary font-weight-bold"
              )
            ),
            div(
              class = "col-6",
              strong("Stop Loss: "),
              br(),
              span(
                paste0("$", signal$entry_bands$stop_loss),
                class = "text-danger font-weight-bold"
              )
            )
          )
        )
      },
      
      # Targets
      if (!is.null(signal$entry_bands)) {
        div(
          class = "mt-3",
          h5("Targets"),
          div(
            class = "row",
            div(
              class = "col-4",
              strong("1 Hour:"),
              br(),
              span(paste0("$", signal$entry_bands$target_1h)),
              br(),
              small(paste0("RR: ", signal$risk_reward$`1h`))
            ),
            div(
              class = "col-4",
              strong("3 Hour:"),
              br(),
              span(paste0("$", signal$entry_bands$target_3h)),
              br(),
              small(paste0("RR: ", signal$risk_reward$`3h`))
            ),
            div(
              class = "col-4",
              strong("EOD:"),
              br(),
              span(paste0("$", signal$entry_bands$target_eod)),
              br(),
              small(paste0("RR: ", signal$risk_reward$eod))
            )
          )
        )
      },
      
      # Market regime and confidence
      div(
        class = "mt-3",
        div(
          class = "row",
          div(
            class = "col-6",
            strong("Market Regime: "),
            span(
              signal$market_regime$label,
              class = paste("badge badge", 
                if(signal$market_regime$label == "trending") "primary" 
                else if(signal$market_regime$label == "ranging") "warning" 
                else "secondary"
              )
            )
          ),
          div(
            class = "col-6",
            strong("Confidence: "),
            span(
              paste0(round(signal$confidence * 100), "%"),
              class = paste("badge badge", conf_color)
            )
          )
        )
      ),
      
      # Model agreement
      div(
        class = "mt-2",
        strong("Model Agreement: "),
        span(signal$model_agreement)
      ),
      
      # Volume ratio
      div(
        class = "mt-2",
        strong("Volume Ratio: "),
        span(
          paste0(signal$volume_ratio, "x"),
          class = if(signal$volume_ratio > 1.5) "text-success" else ""
        )
      ),
      
      # Expiration
      div(
        class = "mt-2 text-muted",
        small(paste("Expires:", format(as.POSIXct(signal$expires_at), "%H:%M")))
      )
    )
  )
  
  return(card_html)
}

# Create active signals table
create_signals_table <- function(signals) {
  if (is.null(signals) || length(signals) == 0) {
    return(data.frame())
  }
  
  # Filter for active signals only
  active_signals <- signals[signals$signal_type != "NEUTRAL", ]
  
  if (nrow(active_signals) == 0) {
    return(data.frame())
  }
  
  # Create table data
  table_data <- data.frame(
    Ticker = active_signals$ticker,
    Signal = active_signals$signal_type,
    Confidence = paste0(round(active_signals$confidence * 100), "%"),
    `Entry Range` = ifelse(
      !is.null(active_signals$entry_bands),
      paste0("$", active_signals$entry_bands$entry_low, "-", active_signals$entry_bands$entry_high),
      "-"
    ),
    `Stop Loss` = ifelse(
      !is.null(active_signals$entry_bands),
      paste0("$", active_signals$entry_bands$stop_loss),
      "-"
    ),
    `Target (1h)` = ifelse(
      !is.null(active_signals$entry_bands),
      paste0("$", active_signals$entry_bands$target_1h),
      "-"
    ),
    Regime = active_signals$market_regime$label,
    `Vol Ratio` = active_signals$volume_ratio,
    stringsAsFactors = FALSE
  )
  
  return(table_data)
}

# Calculate position size based on account value and confidence
calculate_position_size <- function(account_value, signal, risk_per_trade = 0.02) {
  if (is.null(signal) || signal$signal_type == "NEUTRAL") {
    return(list(
      shares = 0,
      position_value = 0,
      risk_amount = 0
    ))
  }
  
  # Calculate risk amount
  risk_amount <- account_value * risk_per_trade
  
  # Calculate position size based on stop distance
  entry_price <- (signal$entry_bands$entry_low + signal$entry_bands$entry_high) / 2
  stop_distance <- abs(entry_price - signal$entry_bands$stop_loss)
  
  # Base shares from risk
  base_shares <- risk_amount / stop_distance
  
  # Adjust by confidence
  adjusted_shares <- floor(base_shares * signal$confidence)
  
  # Calculate position value
  position_value <- adjusted_shares * entry_price
  
  # Ensure position doesn't exceed 25% of account
  max_position <- account_value * 0.25
  if (position_value > max_position) {
    adjusted_shares <- floor(max_position / entry_price)
    position_value <- adjusted_shares * entry_price
  }
  
  return(list(
    shares = adjusted_shares,
    position_value = round(position_value, 2),
    risk_amount = round(adjusted_shares * stop_distance, 2),
    entry_price = round(entry_price, 2),
    stop_distance = round(stop_distance, 2)
  ))
}

# Create market regime overview
create_regime_overview <- function(signals) {
  if (is.null(signals) || length(signals) == 0) {
    return(NULL)
  }
  
  # Count regimes
  regime_counts <- table(sapply(signals, function(s) s$market_regime$label))
  
  # Create overview div
  overview <- div(
    class = "card",
    div(
      class = "card-header",
      h4("Market Regime Overview")
    ),
    div(
      class = "card-body",
      div(
        class = "row",
        lapply(names(regime_counts), function(regime) {
          div(
            class = "col-4 text-center",
            h5(
              regime,
              class = paste("badge badge",
                if(regime == "trending") "primary"
                else if(regime == "ranging") "warning"
                else "secondary"
              )
            ),
            p(paste(regime_counts[regime], "tickers"))
          )
        })
      )
    )
  )
  
  return(overview)
}

# Create alert for high confidence signals
create_signal_alerts <- function(signals, min_confidence = 0.7) {
  if (is.null(signals) || length(signals) == 0) {
    return(NULL)
  }
  
  # Filter high confidence signals
  high_conf_signals <- signals[
    signals$signal_type != "NEUTRAL" & signals$confidence >= min_confidence,
  ]
  
  if (length(high_conf_signals) == 0) {
    return(NULL)
  }
  
  alerts <- lapply(high_conf_signals, function(signal) {
    div(
      class = paste("alert", 
        if(signal$signal_type == "BUY") "alert-success" else "alert-danger",
        "alert-dismissible fade show"
      ),
      role = "alert",
      strong(paste0("High Confidence ", signal$signal_type, ": ", signal$ticker)),
      br(),
      paste0(
        "Entry: $", signal$entry_bands$entry_low, "-$", signal$entry_bands$entry_high,
        " | Stop: $", signal$entry_bands$stop_loss,
        " | Confidence: ", round(signal$confidence * 100), "%"
      ),
      button(
        type = "button",
        class = "close",
        `data-dismiss` = "alert",
        span(HTML("&times;"))
      )
    )
  })
  
  return(alerts)
}

# Format signal summary for display
format_signal_summary <- function(summary) {
  if (is.null(summary)) {
    return(NULL)
  }
  
  summary_div <- div(
    class = "card bg-light",
    div(
      class = "card-body",
      h5("Signal Summary"),
      div(
        class = "row text-center",
        div(
          class = "col-3",
          h3(summary$total_signals),
          p("Total Signals")
        ),
        div(
          class = "col-3",
          h3(summary$buy_signals, class = "text-success"),
          p("Buy Signals")
        ),
        div(
          class = "col-3",
          h3(summary$sell_signals, class = "text-danger"),
          p("Sell Signals")
        ),
        div(
          class = "col-3",
          h3(summary$high_confidence, class = "text-primary"),
          p("High Confidence")
        )
      )
    )
  )
  
  return(summary_div)
}