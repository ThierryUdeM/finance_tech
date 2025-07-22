# Bitcoin Monitor Shiny Dashboard - Azure Edition (Advanced)
# Uses AzureStor package for proper Azure integration

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(jsonlite)
library(AzureStor)
library(lubridate)
library(dplyr)

# Initialize Azure connection
init_azure <- function() {
  # Load credentials
  if (file.exists("../config/.env")) {
    readRenviron("../config/.env")
  } else if (file.exists("config/.env")) {
    readRenviron("config/.env")
  }
  
  # Create blob endpoint
  blob_endpoint <- blob_endpoint(
    endpoint = sprintf("https://%s.blob.core.windows.net", 
                      Sys.getenv("AZURE_STORAGE_ACCOUNT")),
    key = Sys.getenv("AZURE_STORAGE_KEY")
  )
  
  # Get container
  storage_container(blob_endpoint, Sys.getenv("AZURE_CONTAINER_NAME"))
}

# Function to get latest prediction
get_latest_prediction <- function(container) {
  # Try current and previous hours - use UTC time since predictions are stored in UTC
  current_time <- with_tz(Sys.time(), "UTC")
  print(paste("Looking for predictions at:", format(current_time, "%Y-%m-%d %H:%M:%S UTC")))
  
  for (h_offset in 0:5) {
    check_time <- current_time - hours(h_offset)
    blob_path <- sprintf("predictions/%s/%s.json", 
                        format(check_time, "%Y-%m-%d"),
                        format(check_time, "%H"))
    print(paste("Checking blob path:", blob_path))
    
    tryCatch({
      # Download blob content
      temp_file <- tempfile()
      storage_download(container, blob_path, temp_file)
      data <- fromJSON(temp_file)
      unlink(temp_file)
      print(paste("Found prediction! Price:", data$price, "Recommendation:", data$recommendation))
      return(data)
    }, error = function(e) {
      print(paste("  Not found:", e$message))
      NULL
    })
  }
  
  print("No predictions found in last 6 hours")
  NULL
}

# Function to get recent predictions (last 24 hours)
get_recent_predictions <- function(container, hours = 24) {
  predictions <- list()
  current_time <- with_tz(Sys.time(), "UTC")
  
  for (h in 0:(hours-1)) {
    check_time <- current_time - hours(h)
    blob_path <- sprintf("predictions/%s/%s.json", 
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

# Function to get evaluations
get_evaluations <- function(container, days = 7) {
  evaluations <- list()
  
  # List blobs in evaluations folder
  tryCatch({
    blobs <- list_blobs(container, prefix = "evaluations/")
    
    if (!is.null(blobs) && nrow(blobs) > 0) {
      # Filter by date
      cutoff_date <- Sys.Date() - days
      
      for (i in 1:nrow(blobs)) {
        blob_name <- blobs$name[i]
        # Extract date from path
        if (grepl("evaluations/(\\d{4}-\\d{2}-\\d{2})", blob_name)) {
          blob_date <- as.Date(sub("evaluations/(\\d{4}-\\d{2}-\\d{2}).*", "\\1", blob_name))
          
          if (blob_date >= cutoff_date) {
            tryCatch({
              temp_file <- tempfile()
              storage_download(container, blob_name, temp_file)
              data <- fromJSON(temp_file)
              unlink(temp_file)
              evaluations[[length(evaluations) + 1]] <- data
            }, error = function(e) {
              # Skip on error
            })
          }
        }
      }
    }
  }, error = function(e) {
    # If listing fails, return empty list
    # If listing fails, return empty list
    print(paste("Failed to list evaluations:", e$message))
  })
  
  evaluations
}

# Function to get latest performance report
get_latest_report <- function(container) {
  tryCatch({
    # List reports
    print("Listing reports from Azure...")
    reports <- list_blobs(container, prefix = "reports/")
    
    if (!is.null(reports) && nrow(reports) > 0) {
      print(paste("Found", nrow(reports), "reports"))
      print(paste("Report columns:", paste(names(reports), collapse=", ")))
      # Get most recent - sort by last_modified
      if ("last_modified" %in% names(reports)) {
        reports <- reports[order(reports$last_modified, decreasing = TRUE), ]
      } else if ("Last-Modified" %in% names(reports)) {
        reports <- reports[order(reports$`Last-Modified`, decreasing = TRUE), ]
      } else {
        print("Warning: No last_modified column found, using first report")
      }
      latest_name <- reports$name[1]
      print(paste("Loading latest report:", latest_name))
      
      tryCatch({
        temp_file <- tempfile()
        storage_download(container, latest_name, temp_file)
        data <- fromJSON(temp_file)
        unlink(temp_file)
        print("Report loaded successfully")
        print(paste("Report contents:", paste(names(data), collapse=", ")))
        return(data)
      }, error = function(e) {
        print(paste("Error loading report:", e$message))
        NULL
      })
    } else {
      print("No reports found in Azure container")
    }
  }, error = function(e) {
    print(paste("Error listing reports:", e$message))
    # If listing fails, return NULL
    NULL
  })
  
  NULL
}

# UI
ui <- dashboardPage(
  dashboardHeader(title = "BTC-USD Monitor (Azure)"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Live Dashboard", tabName = "dashboard", icon = icon("chart-line")),
      menuItem("Performance", tabName = "performance", icon = icon("trophy")),
      menuItem("History", tabName = "history", icon = icon("history")),
      menuItem("Settings", tabName = "settings", icon = icon("cog"))
    ),
    
    br(),
    
    # Connection status
    div(id = "connection_status", style = "padding: 10px;",
        uiOutput("azure_status")
    ),
    
    br(),
    
    # Refresh controls
    actionButton("refresh", "Refresh Now", icon = icon("sync"), 
                 class = "btn-primary btn-block"),
    
    br(),
    
    # Auto-refresh
    checkboxInput("auto_refresh", "Auto-refresh (5 min)", value = TRUE)
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        .small-box {
          border-radius: 5px;
        }
        .info-box-icon {
          border-radius: 5px;
        }
      "))
    ),
    
    tabItems(
      # Dashboard Tab
      tabItem(
        tabName = "dashboard",
        
        # Current Status
        fluidRow(
          valueBoxOutput("price_box"),
          valueBoxOutput("change_box"),
          valueBoxOutput("recommendation_box")
        ),
        
        # Signal Summary
        fluidRow(
          box(
            title = "Current Signals by Timeframe",
            status = "primary",
            solidHeader = TRUE,
            width = 8,
            DT::dataTableOutput("signals_table")
          ),
          box(
            title = "Signal Distribution",
            status = "primary",
            solidHeader = TRUE,
            width = 4,
            plotlyOutput("signal_pie", height = "250px")
          )
        ),
        
        # Recent Predictions Chart
        fluidRow(
          box(
            title = "24-Hour Prediction History",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("predictions_timeline", height = "400px")
          )
        ),
        
        # Last Update Info
        fluidRow(
          box(
            width = 12,
            p(textOutput("last_update"), style = "text-align: center; color: #666;")
          )
        )
      ),
      
      # Performance Tab
      tabItem(
        tabName = "performance",
        
        # Performance Metrics
        fluidRow(
          valueBoxOutput("accuracy_box"),
          valueBoxOutput("total_predictions_box"),
          valueBoxOutput("correct_predictions_box")
        ),
        
        # Accuracy Charts
        fluidRow(
          box(
            title = "Accuracy by Signal Type",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("accuracy_bars", height = "300px")
          ),
          box(
            title = "Performance Over Time",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("performance_timeline", height = "300px")
          )
        ),
        
        # Recent Evaluations
        fluidRow(
          box(
            title = "Recent Evaluation Results",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            DT::dataTableOutput("evaluations_table")
          )
        )
      ),
      
      # History Tab
      tabItem(
        tabName = "history",
        
        # Date selector
        fluidRow(
          box(
            width = 12,
            dateRangeInput("date_range", "Select Date Range:",
                          start = Sys.Date() - 7,
                          end = Sys.Date(),
                          max = Sys.Date())
          )
        ),
        
        # Historical data
        fluidRow(
          box(
            title = "Prediction History",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            DT::dataTableOutput("history_table")
          )
        )
      ),
      
      # Settings Tab
      tabItem(
        tabName = "settings",
        box(
          title = "Azure Configuration",
          status = "warning",
          solidHeader = TRUE,
          width = 12,
          
          h4("Current Configuration:"),
          verbatimTextOutput("config_info"),
          
          br(),
          
          h4("Data Sources:"),
          p("• Predictions: Updated hourly by GitHub Actions"),
          p("• Evaluations: Performance tracking after 1 hour"),
          p("• Reports: Generated weekly on Sundays"),
          
          br(),
          
          h4("Success Criteria:"),
          p("• BUY: Price increase >0.5% within 1 hour"),
          p("• SELL: Price decrease >0.5% within 1 hour"),
          p("• HOLD: Price change ±0.5% within 1 hour")
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Initialize Azure container
  azure_container <- reactive({
    print("Attempting to initialize Azure connection...")
    tryCatch({
      container <- init_azure()
      print("Azure connection successful!")
      container
    }, error = function(e) {
      # showNotification(paste("Azure connection error:", e$message), 
      #                 session = session, type = "error", duration = NULL)
      print(paste("Azure connection error:", e$message))
      NULL
    })
  })
  
  # Reactive values
  values <- reactiveValues(
    latest_prediction = NULL,
    recent_predictions = list(),
    evaluations = list(),
    performance_report = NULL,
    last_update = NULL
  )
  
  # Fetch all data
  fetch_data <- function() {
    container <- azure_container()
    if (is.null(container)) {
      print("Container is NULL - Azure connection failed")
      return()
    }
    
    withProgress(message = 'Fetching data from Azure...', {
      
      setProgress(0.2, detail = "Getting latest prediction")
      values$latest_prediction <- get_latest_prediction(container)
      print(paste("Latest prediction:", 
                  if(!is.null(values$latest_prediction)) 
                    paste("Price:", values$latest_prediction$price, 
                          "Rec:", values$latest_prediction$recommendation) 
                  else "NULL"))
      
      setProgress(0.4, detail = "Loading recent predictions")
      values$recent_predictions <- get_recent_predictions(container, hours = 24)
      print(paste("Recent predictions count:", length(values$recent_predictions)))
      
      setProgress(0.6, detail = "Loading evaluations")
      values$evaluations <- get_evaluations(container, days = 7)
      print(paste("Evaluations count:", length(values$evaluations)))
      
      setProgress(0.8, detail = "Getting performance report")
      values$performance_report <- get_latest_report(container)
      print(paste("Performance report:", 
                  if(!is.null(values$performance_report)) "loaded" else "NULL"))
      
      values$last_update <- Sys.time()
      
      # showNotification("Data updated successfully", session = session, type = "success", duration = 2)
    })
  }
  
  # Initial load
  observe({
    if (!is.null(azure_container()) && is.null(values$latest_prediction)) {
      fetch_data()
    }
  })
  
  # Manual refresh
  observeEvent(input$refresh, {
    fetch_data()
  })
  
  # Auto-refresh
  observe({
    if (input$auto_refresh) {
      invalidateLater(5 * 60 * 1000)  # 5 minutes
      fetch_data()
    }
  })
  
  # Azure status
  output$azure_status <- renderUI({
    if (!is.null(azure_container())) {
      div(
        icon("check-circle", style = "color: #00a65a;"),
        "Azure Connected",
        style = "color: #00a65a; font-weight: bold;"
      )
    } else {
      div(
        icon("exclamation-circle", style = "color: #dd4b39;"),
        "Azure Disconnected",
        style = "color: #dd4b39; font-weight: bold;"
      )
    }
  })
  
  # Dashboard outputs
  output$price_box <- renderValueBox({
    valueBox(
      value = if (!is.null(values$latest_prediction)) {
        paste0("$", format(values$latest_prediction$price, big.mark = ",", digits = 2, nsmall = 2))
      } else "Loading...",
      subtitle = "BTC-USD Price",
      icon = icon("bitcoin"),
      color = "blue"
    )
  })
  
  output$change_box <- renderValueBox({
    # Calculate 24h change from predictions
    if (length(values$recent_predictions) >= 2) {
      oldest <- values$recent_predictions[[length(values$recent_predictions)]]
      newest <- values$recent_predictions[[1]]
      change <- ((newest$price - oldest$price) / oldest$price) * 100
      
      valueBox(
        value = sprintf("%+.2f%%", change),
        subtitle = "24h Change",
        icon = icon(ifelse(change >= 0, "arrow-up", "arrow-down")),
        color = ifelse(change >= 0, "green", "red")
      )
    } else {
      valueBox(
        value = "...",
        subtitle = "24h Change",
        icon = icon("clock"),
        color = "light-blue"
      )
    }
  })
  
  output$recommendation_box <- renderValueBox({
    if (!is.null(values$latest_prediction)) {
      rec <- values$latest_prediction$recommendation
      color <- switch(rec,
                     "STRONG BUY" = "green",
                     "BUY" = "olive",
                     "STRONG SELL" = "red",
                     "SELL" = "maroon",
                     "yellow")
      
      valueBox(
        value = rec,
        subtitle = sprintf("Buy: %d | Sell: %d", 
                          values$latest_prediction$total_buy_signals,
                          values$latest_prediction$total_sell_signals),
        icon = icon("lightbulb"),
        color = color
      )
    } else {
      valueBox(
        value = "Loading...",
        subtitle = "Recommendation",
        icon = icon("spinner"),
        color = "light-blue"
      )
    }
  })
  
  # Signals table
  output$signals_table <- DT::renderDataTable({
    if (!is.null(values$latest_prediction) && !is.null(values$latest_prediction$intervals)) {
      df <- do.call(rbind, lapply(names(values$latest_prediction$intervals), function(interval) {
        data <- values$latest_prediction$intervals[[interval]]
        data.frame(
          Timeframe = interval,
          Buy = data$buy_signals,
          Sell = data$sell_signals,
          Signal = data$signal,
          Confidence = sprintf("%.3f", data$avg_confidence),
          stringsAsFactors = FALSE
        )
      }))
      
      # Order by timeframe
      tf_order <- c("15m", "1h", "4h", "1d")
      df <- df[order(match(df$Timeframe, tf_order)), ]
      
      DT::datatable(df, 
                    options = list(pageLength = 4, dom = 't'),
                    rownames = FALSE) %>%
        DT::formatStyle("Signal",
                       backgroundColor = DT::styleEqual(
                         c("BUY", "SELL", "NEUTRAL"),
                         c("#d4edda", "#f8d7da", "#fff3cd")
                       ),
                       fontWeight = "bold")
    }
  }, server = FALSE)
  
  # Signal pie chart
  output$signal_pie <- renderPlotly({
    if (!is.null(values$latest_prediction)) {
      df <- data.frame(
        Type = c("Buy", "Sell"),
        Count = c(values$latest_prediction$total_buy_signals,
                 values$latest_prediction$total_sell_signals)
      )
      
      plot_ly(df, labels = ~Type, values = ~Count, type = 'pie',
              marker = list(colors = c('#00a65a', '#dd4b39'))) %>%
        layout(showlegend = TRUE,
               margin = list(l = 0, r = 0, t = 0, b = 0))
    }
  })
  
  # Predictions timeline
  output$predictions_timeline <- renderPlotly({
    if (length(values$recent_predictions) > 0) {
      df <- do.call(rbind, lapply(values$recent_predictions, function(p) {
        data.frame(
          Time = as.POSIXct(p$timestamp),
          Price = p$price,
          Recommendation = p$recommendation,
          stringsAsFactors = FALSE
        )
      }))
      
      df <- df[order(df$Time), ]
      
      # Color by recommendation
      colors <- ifelse(grepl("BUY", df$Recommendation), "#00a65a",
                      ifelse(grepl("SELL", df$Recommendation), "#dd4b39", "#f39c12"))
      
      plot_ly(df, x = ~Time, y = ~Price, type = 'scatter', mode = 'lines+markers',
              line = list(color = '#3c8dbc', width = 2),
              marker = list(size = 8, color = colors),
              text = ~paste("Time:", Time, "<br>Price: $", 
                           format(Price, big.mark = ",", digits = 2),
                           "<br>Rec:", Recommendation),
              hoverinfo = 'text') %>%
        layout(xaxis = list(title = ''),
               yaxis = list(title = 'Price (USD)'),
               showlegend = FALSE)
    }
  })
  
  # Performance outputs
  output$accuracy_box <- renderValueBox({
    if (!is.null(values$performance_report)) {
      accuracy <- values$performance_report$overall_accuracy
      valueBox(
        value = sprintf("%.1f%%", accuracy),
        subtitle = "Overall Accuracy",
        icon = icon("bullseye"),
        color = ifelse(accuracy >= 60, "green", 
                      ifelse(accuracy >= 50, "yellow", "red"))
      )
    } else {
      valueBox(
        value = "...",
        subtitle = "Overall Accuracy",
        icon = icon("chart-line"),
        color = "light-blue"
      )
    }
  })
  
  output$total_predictions_box <- renderValueBox({
    valueBox(
      value = if (!is.null(values$performance_report)) {
        values$performance_report$total_predictions
      } else "...",
      subtitle = "Total Predictions",
      icon = icon("list"),
      color = "blue"
    )
  })
  
  output$correct_predictions_box <- renderValueBox({
    valueBox(
      value = if (!is.null(values$performance_report)) {
        values$performance_report$correct_predictions
      } else "...",
      subtitle = "Correct Predictions",
      icon = icon("check"),
      color = "green"
    )
  })
  
  # Accuracy bars
  output$accuracy_bars <- renderPlotly({
    if (!is.null(values$performance_report)) {
      df <- data.frame(
        Type = c("BUY", "SELL", "HOLD", "OVERALL"),
        Accuracy = c(
          values$performance_report$buy_accuracy,
          values$performance_report$sell_accuracy,
          values$performance_report$hold_accuracy,
          values$performance_report$overall_accuracy
        )
      )
      
      plot_ly(df, x = ~Type, y = ~Accuracy, type = 'bar',
              marker = list(color = c('#00a65a', '#dd4b39', '#f39c12', '#3c8dbc'))) %>%
        layout(yaxis = list(title = 'Accuracy %', range = c(0, 100)),
               xaxis = list(title = ''))
    }
  })
  
  # Performance timeline
  output$performance_timeline <- renderPlotly({
    if (length(values$evaluations) > 0) {
      # Create a timeline of accuracy over time
      df <- do.call(rbind, lapply(values$evaluations, function(e) {
        # Use prediction_time if evaluation_time doesn't exist
        time_field <- if (!is.null(e$evaluation_time)) e$evaluation_time else e$prediction_time
        
        data.frame(
          Time = as.POSIXct(time_field),
          Correct = ifelse(e$was_correct, 1, 0),
          Recommendation = if (!is.null(e$recommendation)) e$recommendation else "Unknown",
          stringsAsFactors = FALSE
        )
      }))
      
      # Sort by time
      df <- df[order(df$Time), ]
      
      # Calculate rolling accuracy (last 10 predictions)
      df$RollingAccuracy <- NA
      if (nrow(df) >= 10) {
        for (i in 10:nrow(df)) {
          df$RollingAccuracy[i] <- mean(df$Correct[(i-9):i]) * 100
        }
      }
      
      # Plot
      plot_ly(df[!is.na(df$RollingAccuracy), ], x = ~Time, y = ~RollingAccuracy, 
              type = 'scatter', mode = 'lines+markers',
              line = list(color = '#3c8dbc', width = 2),
              marker = list(size = 6)) %>%
        layout(yaxis = list(title = 'Rolling Accuracy %', range = c(0, 100)),
               xaxis = list(title = 'Time'),
               hovermode = 'x unified')
    }
  })
  
  # Evaluations table
  output$evaluations_table <- DT::renderDataTable({
    if (length(values$evaluations) > 0) {
      df <- do.call(rbind, lapply(values$evaluations, function(e) {
        data.frame(
          Time = format(as.POSIXct(e$prediction_time), "%m-%d %H:%M"),
          Recommendation = e$recommendation,
          `Past Price` = sprintf("$%.0f", e$past_price),
          `Current Price` = sprintf("$%.0f", e$current_price),
          `Change %` = sprintf("%+.2f%%", e$price_change_pct),
          Result = ifelse(e$was_correct, "✓", "✗"),
          stringsAsFactors = FALSE
        )
      }))
      
      # Sort by time
      df <- df[order(df$Time, decreasing = TRUE), ]
      
      DT::datatable(df, 
                    options = list(pageLength = 10),
                    rownames = FALSE) %>%
        DT::formatStyle("Result",
                       color = DT::styleEqual(c("✓", "✗"), c("#00a65a", "#dd4b39")),
                       fontWeight = "bold")
    }
  })
  
  # History table
  output$history_table <- DT::renderDataTable({
    if (length(values$recent_predictions) > 0) {
      # Filter by date range
      df <- do.call(rbind, lapply(values$recent_predictions, function(p) {
        pred_date <- as.Date(as.POSIXct(p$timestamp))
        if (pred_date >= input$date_range[1] && pred_date <= input$date_range[2]) {
          data.frame(
            Timestamp = format(as.POSIXct(p$timestamp), "%Y-%m-%d %H:%M"),
            Price = paste0("$", format(p$price, big.mark = ",", digits = 2, nsmall = 2)),
            Buy = p$total_buy_signals,
            Sell = p$total_sell_signals,
            Recommendation = p$recommendation,
            stringsAsFactors = FALSE
          )
        }
      }))
      
      if (!is.null(df) && nrow(df) > 0) {
        DT::datatable(df, 
                      options = list(pageLength = 20, order = list(list(0, 'desc'))),
                      rownames = FALSE) %>%
          DT::formatStyle("Recommendation",
                         backgroundColor = DT::styleEqual(
                           c("STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"),
                           c("#00a65a", "#5cb85c", "#f39c12", "#d9534f", "#dd4b39")
                         ),
                         color = "white")
      }
    }
  })
  
  # Last update
  output$last_update <- renderText({
    if (!is.null(values$last_update)) {
      paste("Last updated:", format(values$last_update, "%Y-%m-%d %H:%M:%S"))
    } else {
      "Not yet updated"
    }
  })
  
  # Config info
  output$config_info <- renderPrint({
    list(
      Storage_Account = Sys.getenv("AZURE_STORAGE_ACCOUNT"),
      Container = Sys.getenv("AZURE_CONTAINER_NAME"),
      Connected = !is.null(azure_container())
    )
  })
}

# Run the app
shinyApp(ui = ui, server = server)