# Bitcoin Monitor Shiny Dashboard - Azure Edition
# Displays predictions and performance from Azure Storage

library(shiny)
library(shinydashboard)
library(plotly)
library(DT)
library(jsonlite)
library(httr)
library(lubridate)

# Azure connection helper
azure_connect <- function() {
  # Load credentials from environment or .env file
  if (file.exists("../config/.env")) {
    readRenviron("../config/.env")
  } else if (file.exists("config/.env")) {
    readRenviron("config/.env")
  }
  
  list(
    account_name = Sys.getenv("AZURE_STORAGE_ACCOUNT"),
    access_key = Sys.getenv("AZURE_STORAGE_KEY"),
    container_name = Sys.getenv("AZURE_CONTAINER_NAME")
  )
}

# Function to generate Azure blob URL
get_blob_url <- function(blob_path, azure_config) {
  sprintf("https://%s.blob.core.windows.net/%s/%s",
          azure_config$account_name,
          azure_config$container_name,
          blob_path)
}

# Function to download blob content
download_blob <- function(blob_path, azure_config) {
  url <- get_blob_url(blob_path, azure_config)
  
  # Create signature for authentication
  date_str <- format(Sys.time(), "%a, %d %b %Y %H:%M:%S GMT", tz = "GMT")
  
  # Simplified approach - use SAS token or public access
  # For production, implement proper Azure authentication
  response <- GET(url)
  
  if (status_code(response) == 200) {
    content(response, "text", encoding = "UTF-8")
  } else {
    NULL
  }
}

# Function to list blobs in a directory
list_blobs <- function(prefix, azure_config) {
  # This is a simplified version - in production, use Azure SDK
  # For now, we'll construct paths based on known structure
  NULL
}

# Function to get latest prediction
get_latest_prediction <- function(azure_config) {
  # Get current hour's prediction
  current_time <- Sys.time()
  date_path <- format(current_time, "%Y-%m-%d")
  hour_path <- format(current_time, "%H")
  
  # Try current hour first, then previous hour
  for (h in c(hour_path, sprintf("%02d", as.numeric(hour_path) - 1))) {
    blob_path <- sprintf("predictions/%s/%s.json", date_path, h)
    content <- download_blob(blob_path, azure_config)
    
    if (!is.null(content)) {
      return(fromJSON(content))
    }
  }
  
  # If today's not available, try yesterday
  yesterday <- current_time - days(1)
  date_path <- format(yesterday, "%Y-%m-%d")
  
  for (h in 23:20) {
    blob_path <- sprintf("predictions/%s/%02d.json", date_path, h)
    content <- download_blob(blob_path, azure_config)
    
    if (!is.null(content)) {
      return(fromJSON(content))
    }
  }
  
  NULL
}

# Function to get recent evaluations
get_recent_evaluations <- function(azure_config, days = 1) {
  evaluations <- list()
  
  # Check last N days
  for (d in 0:(days-1)) {
    check_date <- Sys.time() - days(d)
    date_path <- format(check_date, "%Y-%m-%d")
    
    # For simplicity, we'll check specific hours
    # In production, implement proper blob listing
    for (h in 0:23) {
      # Evaluations have timestamps in filenames
      # This is simplified - real implementation would list blobs
      next
    }
  }
  
  evaluations
}

# Function to get performance summary
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
  correct <- sum(sapply(evaluations, function(x) x$was_correct))
  
  buy_evals <- evaluations[sapply(evaluations, function(x) x$recommendation %in% c("BUY", "STRONG BUY"))]
  sell_evals <- evaluations[sapply(evaluations, function(x) x$recommendation %in% c("SELL", "STRONG SELL"))]
  hold_evals <- evaluations[sapply(evaluations, function(x) x$recommendation == "HOLD")]
  
  list(
    total_predictions = total,
    correct_predictions = correct,
    overall_accuracy = if(total > 0) (correct / total * 100) else 0,
    buy_accuracy = if(length(buy_evals) > 0) 
      (sum(sapply(buy_evals, function(x) x$was_correct)) / length(buy_evals) * 100) else 0,
    sell_accuracy = if(length(sell_evals) > 0) 
      (sum(sapply(sell_evals, function(x) x$was_correct)) / length(sell_evals) * 100) else 0,
    hold_accuracy = if(length(hold_evals) > 0) 
      (sum(sapply(hold_evals, function(x) x$was_correct)) / length(hold_evals) * 100) else 0
  )
}

# UI
ui <- dashboardPage(
  dashboardHeader(title = "BTC Predictions - Azure"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard", tabName = "dashboard", icon = icon("chart-line")),
      menuItem("History", tabName = "history", icon = icon("history")),
      menuItem("Performance", tabName = "performance", icon = icon("chart-bar"))
    ),
    
    br(),
    
    # Refresh button
    actionButton("refresh", "Refresh Data", icon = icon("sync"), 
                 class = "btn-primary btn-block"),
    
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
      "))
    ),
    
    tabItems(
      # Dashboard Tab
      tabItem(
        tabName = "dashboard",
        
        # Current Status Row
        fluidRow(
          infoBox(
            "Current Price",
            textOutput("current_price"),
            icon = icon("bitcoin"),
            color = "blue",
            width = 3
          ),
          infoBox(
            "Recommendation",
            textOutput("current_recommendation"),
            icon = icon("lightbulb"),
            color = "yellow",
            width = 3
          ),
          infoBox(
            "Buy Signals",
            textOutput("buy_signals"),
            icon = icon("arrow-up"),
            color = "green",
            width = 3
          ),
          infoBox(
            "Sell Signals",
            textOutput("sell_signals"),
            icon = icon("arrow-down"),
            color = "red",
            width = 3
          )
        ),
        
        # Signals by Timeframe
        fluidRow(
          box(
            title = "Signals by Timeframe",
            status = "primary",
            solidHeader = TRUE,
            width = 8,
            DT::dataTableOutput("timeframe_signals")
          ),
          box(
            title = "Last Update",
            status = "info",
            solidHeader = TRUE,
            width = 4,
            uiOutput("update_info")
          )
        ),
        
        # Recent Predictions Chart
        fluidRow(
          box(
            title = "Recent Predictions",
            status = "primary",
            solidHeader = TRUE,
            width = 12,
            plotlyOutput("predictions_chart", height = "400px")
          )
        )
      ),
      
      # History Tab
      tabItem(
        tabName = "history",
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
      
      # Performance Tab
      tabItem(
        tabName = "performance",
        fluidRow(
          valueBoxOutput("overall_accuracy"),
          valueBoxOutput("total_evaluated"),
          valueBoxOutput("success_rate_trend")
        ),
        
        fluidRow(
          box(
            title = "Accuracy by Signal Type",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            plotlyOutput("accuracy_by_type", height = "300px")
          ),
          box(
            title = "Performance Metrics",
            status = "success",
            solidHeader = TRUE,
            width = 6,
            uiOutput("performance_details")
          )
        ),
        
        fluidRow(
          box(
            title = "Recent Evaluations",
            status = "info",
            solidHeader = TRUE,
            width = 12,
            DT::dataTableOutput("evaluations_table")
          )
        )
      )
    )
  )
)

# Server
server <- function(input, output, session) {
  
  # Azure configuration
  azure_config <- reactive({
    azure_connect()
  })
  
  # Reactive values
  values <- reactiveValues(
    latest_prediction = NULL,
    recent_predictions = list(),
    evaluations = list(),
    performance_summary = NULL,
    last_update = NULL
  )
  
  # Function to fetch all data
  fetch_data <- function() {
    withProgress(message = 'Fetching data from Azure...', {
      
      setProgress(0.2, detail = "Getting latest prediction")
      # Get latest prediction
      values$latest_prediction <- get_latest_prediction(azure_config())
      
      setProgress(0.5, detail = "Loading recent predictions")
      # Get recent predictions (simplified for now)
      # In production, implement proper blob listing
      
      setProgress(0.7, detail = "Loading evaluations")
      # Get recent evaluations
      values$evaluations <- get_recent_evaluations(azure_config(), days = 7)
      
      setProgress(0.9, detail = "Calculating performance")
      # Calculate performance
      values$performance_summary <- calculate_performance_summary(values$evaluations)
      
      values$last_update <- Sys.time()
    })
  }
  
  # Initial data fetch
  observe({
    if (is.null(values$latest_prediction)) {
      fetch_data()
    }
  })
  
  # Manual refresh
  observeEvent(input$refresh, {
    fetch_data()
  })
  
  # Auto-refresh every 5 minutes
  observe({
    invalidateLater(5 * 60 * 1000)
    fetch_data()
  })
  
  # Dashboard outputs
  output$current_price <- renderText({
    if (!is.null(values$latest_prediction)) {
      sprintf("$%,.2f", values$latest_prediction$price)
    } else {
      "Loading..."
    }
  })
  
  output$current_recommendation <- renderText({
    if (!is.null(values$latest_prediction)) {
      values$latest_prediction$recommendation
    } else {
      "Loading..."
    }
  })
  
  output$buy_signals <- renderText({
    if (!is.null(values$latest_prediction)) {
      values$latest_prediction$total_buy_signals
    } else {
      "-"
    }
  })
  
  output$sell_signals <- renderText({
    if (!is.null(values$latest_prediction)) {
      values$latest_prediction$total_sell_signals
    } else {
      "-"
    }
  })
  
  # Timeframe signals table
  output$timeframe_signals <- DT::renderDataTable({
    if (!is.null(values$latest_prediction) && !is.null(values$latest_prediction$intervals)) {
      df <- do.call(rbind, lapply(names(values$latest_prediction$intervals), function(interval) {
        data <- values$latest_prediction$intervals[[interval]]
        data.frame(
          Interval = interval,
          Buy = data$buy_signals,
          Sell = data$sell_signals,
          Signal = data$signal,
          Confidence = sprintf("%.2f", data$avg_confidence),
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
  
  # Update info
  output$update_info <- renderUI({
    if (!is.null(values$latest_prediction)) {
      tagList(
        h4("Prediction Time:"),
        p(values$latest_prediction$timestamp),
        hr(),
        h4("Data Source:"),
        p("Azure Storage"),
        p(azure_config()$container_name),
        hr(),
        p("Next update in:", 
          textOutput("time_to_update", inline = TRUE))
      )
    }
  })
  
  # Last update times
  output$last_update_sidebar <- renderText({
    if (!is.null(values$last_update)) {
      paste("Updated:", format(values$last_update, "%H:%M:%S"))
    }
  })
  
  # Performance outputs
  output$overall_accuracy <- renderValueBox({
    valueBox(
      value = if (!is.null(values$performance_summary)) {
        sprintf("%.1f%%", values$performance_summary$overall_accuracy)
      } else "...",
      subtitle = "Overall Accuracy",
      icon = icon("bullseye"),
      color = if (!is.null(values$performance_summary)) {
        if (values$performance_summary$overall_accuracy >= 60) "green"
        else if (values$performance_summary$overall_accuracy >= 50) "yellow"
        else "red"
      } else "blue"
    )
  })
  
  output$total_evaluated <- renderValueBox({
    valueBox(
      value = if (!is.null(values$performance_summary)) {
        values$performance_summary$total_predictions
      } else "...",
      subtitle = "Total Evaluated",
      icon = icon("check-circle"),
      color = "blue"
    )
  })
  
  output$success_rate_trend <- renderValueBox({
    # Calculate trend (simplified)
    trend <- "stable"
    valueBox(
      value = trend,
      subtitle = "Trend",
      icon = icon(switch(trend,
                        "improving" = "arrow-up",
                        "declining" = "arrow-down",
                        "chart-line")),
      color = "purple"
    )
  })
  
  # Accuracy by type chart
  output$accuracy_by_type <- renderPlotly({
    if (!is.null(values$performance_summary)) {
      df <- data.frame(
        Type = c("BUY", "SELL", "HOLD"),
        Accuracy = c(
          values$performance_summary$buy_accuracy,
          values$performance_summary$sell_accuracy,
          values$performance_summary$hold_accuracy
        )
      )
      
      plot_ly(df, x = ~Type, y = ~Accuracy, type = 'bar',
              marker = list(color = c('#00a65a', '#dd4b39', '#f39c12'))) %>%
        layout(yaxis = list(title = 'Accuracy %', range = c(0, 100)),
               xaxis = list(title = ''))
    }
  })
  
  # Performance details
  output$performance_details <- renderUI({
    if (!is.null(values$performance_summary)) {
      tagList(
        h4("Success Metrics:"),
        p(sprintf("Correct Predictions: %d / %d",
                  values$performance_summary$correct_predictions,
                  values$performance_summary$total_predictions)),
        hr(),
        h5("By Signal Type:"),
        p(sprintf("BUY Accuracy: %.1f%%", values$performance_summary$buy_accuracy),
          class = "signal-buy"),
        p(sprintf("SELL Accuracy: %.1f%%", values$performance_summary$sell_accuracy),
          class = "signal-sell"),
        p(sprintf("HOLD Accuracy: %.1f%%", values$performance_summary$hold_accuracy),
          class = "signal-hold"),
        hr(),
        p("Threshold: ±0.5% in 1 hour", style = "font-size: 12px; color: #666;")
      )
    }
  })
}

# Run the app
shinyApp(ui = ui, server = server)