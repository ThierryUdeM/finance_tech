# Multi-Ticker Monitor Shiny Dashboard - Azure Edition
# Displays predictions and performance for BTC, NVDA, and AC.TO from Azure Storage

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(jsonlite)
library(AzureStor)
library(lubridate)
library(dplyr)
library(tidyr)

# Configuration
TICKERS <- list(
  "BTC-USD" = list(name = "Bitcoin", icon = "bitcoin", color = "orange"),
  "NVDA" = list(name = "NVIDIA", icon = "microchip", color = "green"),
  "AC.TO" = list(name = "Air Canada", icon = "plane", color = "red")
)

# Initialize Azure connection
init_azure <- function() {
  # Load credentials
  if (file.exists("../config/.env")) {
    readRenviron("../config/.env")
  } else if (file.exists("config/.env")) {
    readRenviron("config/.env")
  }
  
  # Handle both naming conventions
  storage_account <- Sys.getenv("STORAGE_ACCOUNT_NAME")
  if (nchar(storage_account) == 0) {
    storage_account <- Sys.getenv("AZURE_STORAGE_ACCOUNT")
  }
  
  access_key <- Sys.getenv("ACCESS_KEY")
  if (nchar(access_key) == 0) {
    access_key <- Sys.getenv("AZURE_STORAGE_KEY")
  }
  
  container_name <- Sys.getenv("CONTAINER_NAME")
  if (nchar(container_name) == 0) {
    container_name <- Sys.getenv("AZURE_CONTAINER_NAME")
  }
  
  # Create blob endpoint
  tryCatch({
    blob_endpoint <- blob_endpoint(
      endpoint = sprintf("https://%s.blob.core.windows.net", storage_account),
      key = access_key
    )
    
    # Get container
    storage_container(blob_endpoint, container_name)
  }, error = function(e) {
    message("Azure connection error: ", e$message)
    NULL
  })
}

# Function to get latest prediction for a ticker
get_latest_prediction <- function(container, ticker) {
  if (is.null(container)) return(NULL)
  
  # Try current and previous hours - use UTC time
  current_time <- with_tz(Sys.time(), "UTC")
  
  for (h_offset in 0:5) {
    check_time <- current_time - hours(h_offset)
    blob_path <- sprintf("predictions/%s/%s/%s.json", 
                        ticker,
                        format(check_time, "%Y-%m-%d"),
                        format(check_time, "%H"))
    
    tryCatch({
      temp_file <- tempfile()
      storage_download(container, blob_path, temp_file)
      data <- fromJSON(temp_file)
      unlink(temp_file)
      return(data)
    }, error = function(e) {
      NULL
    })
  }
  NULL
}

# Function to get recent predictions for a ticker
get_recent_predictions <- function(container, ticker, hours = 24) {
  if (is.null(container)) return(list())
  
  predictions <- list()
  current_time <- with_tz(Sys.time(), "UTC")
  
  for (h in 0:(hours-1)) {
    check_time <- current_time - hours(h)
    blob_path <- sprintf("predictions/%s/%s/%s.json", 
                        ticker,
                        format(check_time, "%Y-%m-%d"),
                        format(check_time, "%H"))
    
    tryCatch({
      temp_file <- tempfile()
      storage_download(container, blob_path, temp_file)
      data <- fromJSON(temp_file)
      unlink(temp_file)
      predictions[[length(predictions) + 1]] <- data
    }, error = function(e) {
      # Skip if not found
    })
  }
  
  predictions
}

# Function to get evaluations for a ticker
get_evaluations <- function(container, ticker, days = 7) {
  if (is.null(container)) return(list())
  
  evaluations <- list()
  current_time <- with_tz(Sys.time(), "UTC")
  
  # Check each day
  for (d in 0:(days-1)) {
    check_date <- current_time - days(d)
    date_str <- format(check_date, "%Y-%m-%d")
    
    # Try to list blobs for this day
    prefix <- sprintf("evaluations/%s/%s/", ticker, date_str)
    
    tryCatch({
      blobs <- list_blobs(container, prefix = prefix)
      
      if (!is.null(blobs) && nrow(blobs) > 0) {
        for (i in 1:nrow(blobs)) {
          tryCatch({
            temp_file <- tempfile()
            storage_download(container, blobs$name[i], temp_file)
            data <- fromJSON(temp_file)
            unlink(temp_file)
            evaluations[[length(evaluations) + 1]] <- data
          }, error = function(e) {
            # Skip if error
          })
        }
      }
    }, error = function(e) {
      # Skip if error
    })
  }
  
  evaluations
}

# Function to calculate performance summary
calculate_performance_summary <- function(evaluations) {
  if (length(evaluations) == 0) {
    return(list(
      total_predictions = 0,
      correct_predictions = 0,
      overall_accuracy = 0,
      buy_accuracy = 0,
      sell_accuracy = 0,
      hold_accuracy = 0
    ))
  }
  
  # Calculate metrics
  total <- length(evaluations)
  correct <- sum(sapply(evaluations, function(x) isTRUE(x$was_correct)))
  
  buy_evals <- evaluations[sapply(evaluations, function(x) x$recommendation %in% c("BUY", "STRONG BUY"))]
  sell_evals <- evaluations[sapply(evaluations, function(x) x$recommendation %in% c("SELL", "STRONG SELL"))]
  hold_evals <- evaluations[sapply(evaluations, function(x) x$recommendation == "HOLD")]
  
  list(
    total_predictions = total,
    correct_predictions = correct,
    overall_accuracy = if(total > 0) (correct / total * 100) else 0,
    buy_accuracy = if(length(buy_evals) > 0) 
      (sum(sapply(buy_evals, function(x) isTRUE(x$was_correct))) / length(buy_evals) * 100) else 0,
    sell_accuracy = if(length(sell_evals) > 0) 
      (sum(sapply(sell_evals, function(x) isTRUE(x$was_correct))) / length(sell_evals) * 100) else 0,
    hold_accuracy = if(length(hold_evals) > 0) 
      (sum(sapply(hold_evals, function(x) isTRUE(x$was_correct))) / length(hold_evals) * 100) else 0
  )
}

# Function to create a ticker dashboard content
create_ticker_dashboard <- function(ticker_id, ticker_data) {
  tagList(
    # Current Status Row
    fluidRow(
      infoBox(
        "Current Price",
        textOutput(paste0("price_", ticker_id)),
        icon = icon(ticker_data$icon),
        color = ticker_data$color,
        width = 3
      ),
      infoBox(
        "YOLO Recommendation",
        textOutput(paste0("recommendation_", ticker_id)),
        icon = icon("lightbulb"),
        color = "yellow",
        width = 3
      ),
      infoBox(
        "Buy Signals",
        textOutput(paste0("buy_signals_", ticker_id)),
        icon = icon("arrow-up"),
        color = "green",
        width = 3
      ),
      infoBox(
        "Sell Signals",
        textOutput(paste0("sell_signals_", ticker_id)),
        icon = icon("arrow-down"),
        color = "red",
        width = 3
      )
    ),
    
    # Signals and Performance Row
    fluidRow(
      box(
        title = "Signals by Timeframe",
        status = "primary",
        solidHeader = TRUE,
        width = 6,
        DT::dataTableOutput(paste0("timeframe_signals_", ticker_id))
      ),
      box(
        title = "Performance Summary",
        status = "success",
        solidHeader = TRUE,
        width = 6,
        plotlyOutput(paste0("performance_gauge_", ticker_id), height = "300px")
      )
    ),
    
    # Recent Predictions Chart
    fluidRow(
      box(
        title = "24-Hour Prediction History",
        status = "primary",
        solidHeader = TRUE,
        width = 12,
        plotlyOutput(paste0("predictions_chart_", ticker_id), height = "400px")
      )
    ),
    
    # Performance Metrics
    fluidRow(
      box(
        title = "Accuracy by Signal Type",
        status = "info",
        solidHeader = TRUE,
        width = 6,
        plotlyOutput(paste0("accuracy_chart_", ticker_id), height = "300px")
      ),
      box(
        title = "Detailed Metrics",
        status = "info",
        solidHeader = TRUE,
        width = 6,
        uiOutput(paste0("metrics_", ticker_id))
      )
    )
  )
}

# UI
ui <- dashboardPage(
  dashboardHeader(
    title = "Multi-Ticker YOLO Monitor",
    tags$li(
      class = "dropdown",
      tags$a(
        href = "#",
        class = "dropdown-toggle",
        `data-toggle` = "dropdown",
        icon("bell"),
        tags$span(class = "label label-warning", textOutput("alert_count", inline = TRUE))
      )
    )
  ),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("BTC-USD", tabName = "btc", icon = icon("bitcoin")),
      menuItem("NVIDIA", tabName = "nvda", icon = icon("microchip")),
      menuItem("Air Canada", tabName = "ac", icon = icon("plane")),
      menuItem("Compare", tabName = "compare", icon = icon("chart-line")),
      menuItem("Settings", tabName = "settings", icon = icon("cog"))
    ),
    
    br(),
    
    # Refresh button
    actionButton("refresh", "Refresh All", icon = icon("sync"), 
                 class = "btn-primary btn-block"),
    
    br(),
    
    # Auto-refresh toggle
    checkboxInput("auto_refresh", "Auto-refresh (5 min)", value = TRUE),
    
    br(),
    
    # Last update time
    textOutput("last_update_sidebar")
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        .info-box-number {
          font-size: 24px;
          font-weight: bold;
        }
        .signal-buy { color: #00a65a; }
        .signal-sell { color: #dd4b39; }
        .signal-neutral { color: #f39c12; }
        .signal-hold { color: #3c8dbc; }
        .small-box h3 {
          font-size: 28px;
        }
      "))
    ),
    
    tabItems(
      # Overview Tab
      tabItem(
        tabName = "overview",
        h2("Multi-Ticker Overview"),
        
        fluidRow(
          lapply(names(TICKERS), function(ticker) {
            ticker_data <- TICKERS[[ticker]]
            ticker_id <- gsub("[.-]", "_", ticker)
            
            column(4,
              box(
                title = ticker_data$name,
                status = "primary",
                solidHeader = TRUE,
                width = NULL,
                
                # Mini summary
                h4(textOutput(paste0("overview_price_", ticker_id)), style = "margin: 10px 0;"),
                h3(textOutput(paste0("overview_rec_", ticker_id)), 
                   style = paste0("margin: 10px 0; color: ", ticker_data$color, ";")),
                
                # Mini performance gauge
                plotlyOutput(paste0("overview_gauge_", ticker_id), height = "200px"),
                
                # Action button
                actionButton(paste0("goto_", ticker_id), "View Details", 
                            icon = icon("arrow-right"),
                            class = "btn-block btn-sm",
                            style = paste0("background-color: ", ticker_data$color, "; color: white;"))
              )
            )
          })
        ),
        
        # Combined performance chart
        fluidRow(
          box(
            title = "24-Hour Recommendations Comparison",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("combined_chart", height = "400px")
          )
        )
      ),
      
      # Individual ticker tabs
      tabItem(tabName = "btc", create_ticker_dashboard("BTC_USD", TICKERS[["BTC-USD"]])),
      tabItem(tabName = "nvda", create_ticker_dashboard("NVDA", TICKERS[["NVDA"]])),
      tabItem(tabName = "ac", create_ticker_dashboard("AC_TO", TICKERS[["AC.TO"]])),
      
      # Compare Tab
      tabItem(
        tabName = "compare",
        h2("Compare Tickers"),
        
        fluidRow(
          box(
            title = "Performance Comparison",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            
            # Comparison table
            DT::dataTableOutput("comparison_table")
          )
        ),
        
        fluidRow(
          box(
            title = "Accuracy Trends",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("accuracy_trends", height = "400px")
          )
        )
      ),
      
      # Settings Tab
      tabItem(
        tabName = "settings",
        h2("Settings"),
        
        fluidRow(
          box(
            title = "Azure Connection Status",
            status = "info",
            solidHeader = TRUE,
            width = 6,
            uiOutput("azure_status")
          ),
          box(
            title = "Data Settings",
            status = "warning",
            solidHeader = TRUE,
            width = 6,
            sliderInput("history_hours", "History Hours:", 
                       min = 6, max = 48, value = 24, step = 6),
            sliderInput("eval_days", "Evaluation Days:", 
                       min = 1, max = 14, value = 7, step = 1),
            actionButton("apply_settings", "Apply", icon = icon("check"))
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Azure configuration
  azure_container <- reactive({
    init_azure()
  })
  
  # Reactive values for each ticker
  ticker_data <- reactiveValues()
  
  # Initialize ticker data structure
  for (ticker in names(TICKERS)) {
    ticker_id <- gsub("[.-]", "_", ticker)
    ticker_data[[ticker_id]] <- list(
      latest_prediction = NULL,
      recent_predictions = list(),
      evaluations = list(),
      performance_summary = NULL
    )
  }
  
  # Global reactive values
  values <- reactiveValues(
    last_update = NULL,
    history_hours = 24,
    eval_days = 7
  )
  
  # Function to fetch data for a ticker
  fetch_ticker_data <- function(ticker) {
    ticker_id <- gsub("[.-]", "_", ticker)
    container <- azure_container()
    
    if (!is.null(container)) {
      # Get latest prediction
      ticker_data[[ticker_id]]$latest_prediction <- get_latest_prediction(container, ticker)
      
      # Get recent predictions
      ticker_data[[ticker_id]]$recent_predictions <- get_recent_predictions(
        container, ticker, values$history_hours
      )
      
      # Get evaluations
      ticker_data[[ticker_id]]$evaluations <- get_evaluations(
        container, ticker, values$eval_days
      )
      
      # Calculate performance
      ticker_data[[ticker_id]]$performance_summary <- calculate_performance_summary(
        ticker_data[[ticker_id]]$evaluations
      )
    }
  }
  
  # Function to fetch all data
  fetch_all_data <- function() {
    withProgress(message = 'Fetching data from Azure...', {
      for (i in seq_along(TICKERS)) {
        ticker <- names(TICKERS)[i]
        setProgress(i / length(TICKERS), detail = paste("Loading", ticker))
        fetch_ticker_data(ticker)
      }
      values$last_update <- Sys.time()
    })
  }
  
  # Initial data fetch
  observe({
    if (is.null(values$last_update)) {
      fetch_all_data()
    }
  })
  
  # Manual refresh
  observeEvent(input$refresh, {
    fetch_all_data()
  })
  
  # Auto-refresh
  observe({
    if (input$auto_refresh) {
      invalidateLater(5 * 60 * 1000)  # 5 minutes
      fetch_all_data()
    }
  })
  
  # Apply settings
  observeEvent(input$apply_settings, {
    values$history_hours <- input$history_hours
    values$eval_days <- input$eval_days
    fetch_all_data()
  })
  
  # Create outputs for each ticker
  for (ticker in names(TICKERS)) {
    local({
      current_ticker <- ticker
      ticker_id <- gsub("[.-]", "_", current_ticker)
      ticker_info <- TICKERS[[current_ticker]]
      
      # Price output
      output[[paste0("price_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          paste0("$", formatC(data$price, format = "f", digits = 2, big.mark = ","))
        } else {
          "Loading..."
        }
      })
      
      # Recommendation output
      output[[paste0("recommendation_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          data$recommendation
        } else {
          "Loading..."
        }
      })
      
      # Buy signals
      output[[paste0("buy_signals_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          as.character(data$total_buy_signals)
        } else {
          "-"
        }
      })
      
      # Sell signals
      output[[paste0("sell_signals_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          as.character(data$total_sell_signals)
        } else {
          "-"
        }
      })
      
      # Timeframe signals table
      output[[paste0("timeframe_signals_", ticker_id)]] <- DT::renderDataTable({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data) && !is.null(data$intervals)) {
          df <- do.call(rbind, lapply(names(data$intervals), function(interval) {
            interval_data <- data$intervals[[interval]]
            data.frame(
              Interval = interval,
              Buy = interval_data$buy_signals,
              Sell = interval_data$sell_signals,
              Signal = interval_data$signal,
              Confidence = sprintf("%.2f", interval_data$avg_confidence),
              stringsAsFactors = FALSE
            )
          }))
          
          DT::datatable(df, 
                       options = list(pageLength = 4, dom = 't'),
                       rownames = FALSE) %>%
            DT::formatStyle("Signal",
                           color = DT::styleEqual(
                             c("BUY", "SELL", "NEUTRAL"),
                             c("#00a65a", "#dd4b39", "#f39c12")
                           ),
                           fontWeight = "bold")
        }
      }, server = FALSE)
      
      # Performance gauge
      output[[paste0("performance_gauge_", ticker_id)]] <- renderPlotly({
        perf <- ticker_data[[ticker_id]]$performance_summary
        if (!is.null(perf)) {
          plot_ly(
            type = "indicator",
            mode = "gauge+number",
            value = perf$overall_accuracy,
            title = list(text = "Overall Accuracy"),
            gauge = list(
              axis = list(range = list(0, 100)),
              bar = list(color = if(perf$overall_accuracy >= 60) "green" 
                        else if(perf$overall_accuracy >= 50) "orange" 
                        else "red"),
              steps = list(
                list(range = c(0, 50), color = "lightgray"),
                list(range = c(50, 60), color = "lightyellow"),
                list(range = c(60, 100), color = "lightgreen")
              )
            )
          ) %>%
            layout(margin = list(l = 20, r = 20, t = 40, b = 20))
        }
      })
      
      # Predictions chart
      output[[paste0("predictions_chart_", ticker_id)]] <- renderPlotly({
        predictions <- ticker_data[[ticker_id]]$recent_predictions
        if (length(predictions) > 0) {
          # Convert to dataframe
          df <- do.call(rbind, lapply(predictions, function(p) {
            data.frame(
              Time = as.POSIXct(p$timestamp),
              Price = p$price,
              Recommendation = p$recommendation,
              BuySignals = p$total_buy_signals,
              SellSignals = p$total_sell_signals,
              stringsAsFactors = FALSE
            )
          }))
          
          df <- df[order(df$Time),]
          
          # Create plot
          plot_ly(df, x = ~Time) %>%
            add_trace(y = ~Price, type = 'scatter', mode = 'lines+markers',
                     name = 'Price', yaxis = 'y2') %>%
            add_trace(y = ~BuySignals, type = 'bar', name = 'Buy Signals',
                     marker = list(color = 'green')) %>%
            add_trace(y = ~SellSignals, type = 'bar', name = 'Sell Signals',
                     marker = list(color = 'red')) %>%
            layout(
              yaxis = list(title = 'Signals'),
              yaxis2 = list(title = 'Price', overlaying = 'y', side = 'right'),
              barmode = 'group',
              hovermode = 'x unified'
            )
        }
      })
      
      # Accuracy chart
      output[[paste0("accuracy_chart_", ticker_id)]] <- renderPlotly({
        perf <- ticker_data[[ticker_id]]$performance_summary
        if (!is.null(perf)) {
          df <- data.frame(
            Type = c("BUY", "SELL", "HOLD"),
            Accuracy = c(perf$buy_accuracy, perf$sell_accuracy, perf$hold_accuracy)
          )
          
          plot_ly(df, x = ~Type, y = ~Accuracy, type = 'bar',
                 marker = list(color = c('#00a65a', '#dd4b39', '#f39c12'))) %>%
            layout(yaxis = list(title = 'Accuracy %', range = c(0, 100)))
        }
      })
      
      # Detailed metrics
      output[[paste0("metrics_", ticker_id)]] <- renderUI({
        perf <- ticker_data[[ticker_id]]$performance_summary
        if (!is.null(perf)) {
          tagList(
            h4("Performance Metrics:"),
            p(sprintf("Total Evaluated: %d", perf$total_predictions)),
            p(sprintf("Correct Predictions: %d", perf$correct_predictions)),
            hr(),
            h5("Accuracy by Type:"),
            p(sprintf("BUY: %.1f%%", perf$buy_accuracy), class = "signal-buy"),
            p(sprintf("SELL: %.1f%%", perf$sell_accuracy), class = "signal-sell"),
            p(sprintf("HOLD: %.1f%%", perf$hold_accuracy), class = "signal-hold")
          )
        }
      })
      
      # Overview outputs
      output[[paste0("overview_price_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          paste0(ticker_info$name, ": $", formatC(data$price, format = "f", digits = 2, big.mark = ","))
        } else {
          paste(ticker_info$name, ": Loading...")
        }
      })
      
      output[[paste0("overview_rec_", ticker_id)]] <- renderText({
        data <- ticker_data[[ticker_id]]$latest_prediction
        if (!is.null(data)) {
          data$recommendation
        } else {
          "-"
        }
      })
      
      output[[paste0("overview_gauge_", ticker_id)]] <- renderPlotly({
        perf <- ticker_data[[ticker_id]]$performance_summary
        if (!is.null(perf)) {
          plot_ly(
            type = "indicator",
            mode = "number+delta",
            value = perf$overall_accuracy,
            delta = list(reference = 50),
            title = list(text = "Accuracy %"),
            domain = list(x = c(0, 1), y = c(0, 1))
          ) %>%
            layout(margin = list(l = 20, r = 20, t = 40, b = 20))
        }
      })
      
      # Navigation buttons
      observeEvent(input[[paste0("goto_", ticker_id)]], {
        updateTabItems(session, "tabs", selected = 
                      switch(current_ticker,
                            "BTC-USD" = "btc",
                            "NVDA" = "nvda",
                            "AC.TO" = "ac"))
      })
    })
  }
  
  # Combined chart
  output$combined_chart <- renderPlotly({
    # Collect all predictions
    all_data <- list()
    
    for (ticker in names(TICKERS)) {
      ticker_id <- gsub("[.-]", "_", ticker)
      predictions <- ticker_data[[ticker_id]]$recent_predictions
      
      if (length(predictions) > 0) {
        df <- do.call(rbind, lapply(predictions, function(p) {
          data.frame(
            Ticker = ticker,
            Time = as.POSIXct(p$timestamp),
            Recommendation = p$recommendation,
            stringsAsFactors = FALSE
          )
        }))
        all_data[[ticker]] <- df
      }
    }
    
    if (length(all_data) > 0) {
      combined_df <- do.call(rbind, all_data)
      
      # Convert recommendations to numeric
      combined_df$RecValue <- ifelse(combined_df$Recommendation == "STRONG BUY", 2,
                                    ifelse(combined_df$Recommendation == "BUY", 1,
                                          ifelse(combined_df$Recommendation == "HOLD", 0,
                                                ifelse(combined_df$Recommendation == "SELL", -1, -2))))
      
      plot_ly(combined_df, x = ~Time, y = ~RecValue, color = ~Ticker,
             type = 'scatter', mode = 'lines+markers') %>%
        layout(
          yaxis = list(
            title = 'Recommendation',
            tickvals = c(-2, -1, 0, 1, 2),
            ticktext = c('STRONG SELL', 'SELL', 'HOLD', 'BUY', 'STRONG BUY')
          ),
          hovermode = 'x unified'
        )
    }
  })
  
  # Comparison table
  output$comparison_table <- DT::renderDataTable({
    # Collect performance data
    comparison_data <- list()
    
    for (ticker in names(TICKERS)) {
      ticker_id <- gsub("[.-]", "_", ticker)
      perf <- ticker_data[[ticker_id]]$performance_summary
      latest <- ticker_data[[ticker_id]]$latest_prediction
      
      if (!is.null(perf) && !is.null(latest)) {
        comparison_data[[ticker]] <- data.frame(
          Ticker = TICKERS[[ticker]]$name,
          `Current Price` = paste0("$", formatC(latest$price, format = "f", digits = 2, big.mark = ",")),
          Recommendation = latest$recommendation,
          `Overall Accuracy` = sprintf("%.1f%%", perf$overall_accuracy),
          `Buy Accuracy` = sprintf("%.1f%%", perf$buy_accuracy),
          `Sell Accuracy` = sprintf("%.1f%%", perf$sell_accuracy),
          `Total Evaluated` = perf$total_predictions,
          stringsAsFactors = FALSE
        )
      }
    }
    
    if (length(comparison_data) > 0) {
      df <- do.call(rbind, comparison_data)
      
      DT::datatable(df, 
                   options = list(pageLength = 10, dom = 't'),
                   rownames = FALSE) %>%
        DT::formatStyle("Recommendation",
                       backgroundColor = DT::styleEqual(
                         c("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"),
                         c("#00a65a", "#5cb85c", "#f0ad4e", "#d9534f", "#dd4b39")
                       ),
                       color = "white",
                       fontWeight = "bold")
    }
  })
  
  # Last update
  output$last_update_sidebar <- renderText({
    if (!is.null(values$last_update)) {
      paste("Updated:", format(values$last_update, "%H:%M:%S"))
    }
  })
  
  # Alert count
  output$alert_count <- renderText({
    # Count strong buy/sell signals
    alerts <- 0
    for (ticker in names(TICKERS)) {
      ticker_id <- gsub("[.-]", "_", ticker)
      latest <- ticker_data[[ticker_id]]$latest_prediction
      if (!is.null(latest) && latest$recommendation %in% c("STRONG BUY", "STRONG SELL")) {
        alerts <- alerts + 1
      }
    }
    as.character(alerts)
  })
  
  # Azure status
  output$azure_status <- renderUI({
    container <- azure_container()
    
    # Get the actual values being used
    storage_account <- Sys.getenv("STORAGE_ACCOUNT_NAME")
    if (nchar(storage_account) == 0) {
      storage_account <- Sys.getenv("AZURE_STORAGE_ACCOUNT")
    }
    
    container_name <- Sys.getenv("CONTAINER_NAME")
    if (nchar(container_name) == 0) {
      container_name <- Sys.getenv("AZURE_CONTAINER_NAME")
    }
    
    if (!is.null(container)) {
      tagList(
        icon("check-circle", style = "color: green;"),
        " Connected to Azure",
        br(),
        br(),
        p("Storage Account:", storage_account),
        p("Container:", container_name)
      )
    } else {
      tagList(
        icon("times-circle", style = "color: red;"),
        " Not connected",
        br(),
        br(),
        p("Check your Azure credentials in config/.env"),
        p("Expected variables: AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER_NAME"),
        p("Or: STORAGE_ACCOUNT_NAME, ACCESS_KEY, CONTAINER_NAME")
      )
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)