#!/usr/bin/env Rscript

# Reddit Trend Analyzer - Identifies trending stocks from accumulated data
# Run this after collecting data for a few days

library(dplyr)
library(readr)
library(lubridate)
library(ggplot2)

# Configuration
DATA_DIR <- "data/traction/history"
dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)

# First, run the scraper and save with timestamp
run_scraper <- function() {
  cat("\n=== Running Reddit Scraper ===\n")
  
  # Run the scraper
  source("reddit_sentiment_strict.R")
  
  # Save timestamped copy
  if(file.exists("data/traction/reddit_mentions_strict.csv")) {
    data <- read_csv("data/traction/reddit_mentions_strict.csv", show_col_types = FALSE)
    data$timestamp <- Sys.time()
    
    # Append to history file
    history_file <- file.path(DATA_DIR, "mention_history.csv")
    
    if(file.exists(history_file)) {
      write_csv(data, history_file, append = TRUE)
    } else {
      write_csv(data, history_file)
    }
    
    cat(sprintf("\n✅ Saved %d tickers to history at %s\n", 
                nrow(data), format(Sys.time(), "%Y-%m-%d %H:%M")))
  }
}

# Analyze trends from accumulated data
analyze_trends <- function(days_back = 7) {
  history_file <- file.path(DATA_DIR, "mention_history.csv")
  
  if(!file.exists(history_file)) {
    cat("\n❌ No historical data found. Run the scraper multiple times first.\n")
    return(NULL)
  }
  
  # Load history
  history <- read_csv(history_file, show_col_types = FALSE)
  
  # Filter to recent period
  cutoff_date <- Sys.time() - days(days_back)
  history <- history %>% filter(timestamp >= cutoff_date)
  
  if(nrow(history) == 0) {
    cat("\n❌ No data in the specified time period.\n")
    return(NULL)
  }
  
  # Calculate daily aggregates
  daily <- history %>%
    mutate(date = as.Date(timestamp)) %>%
    group_by(ticker, date) %>%
    summarise(
      daily_mentions = sum(mentions),
      max_mentions = max(mentions),
      measurements = n(),
      .groups = "drop"
    )
  
  # Calculate trends
  trends <- daily %>%
    group_by(ticker) %>%
    summarise(
      days_active = n_distinct(date),
      total_mentions = sum(daily_mentions),
      avg_daily = mean(daily_mentions),
      max_daily = max(daily_mentions),
      first_seen = min(date),
      last_seen = max(date),
      .groups = "drop"
    ) %>%
    mutate(
      days_since_first = as.numeric(Sys.Date() - first_seen),
      recency_score = 1 / (as.numeric(Sys.Date() - last_seen) + 1)
    )
  
  # Calculate momentum (recent vs earlier mentions)
  momentum <- daily %>%
    mutate(
      period = ifelse(date >= Sys.Date() - days(2), "recent", "earlier")
    ) %>%
    group_by(ticker, period) %>%
    summarise(
      period_mentions = sum(daily_mentions),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(names_from = period, values_from = period_mentions, values_fill = 0) %>%
    mutate(
      momentum = ifelse(earlier > 0, recent / earlier, recent),
      growth_pct = ifelse(earlier > 0, 100 * (recent - earlier) / earlier, 100)
    )
  
  # Combine metrics
  analysis <- trends %>%
    left_join(momentum, by = "ticker") %>%
    mutate(
      # Composite trend score
      trend_score = (total_mentions * 0.3) + 
                   (momentum * 10) + 
                   (recency_score * 20) +
                   (days_active * 2),
      
      # Categorize
      category = case_when(
        days_since_first <= 1 & total_mentions >= 5 ~ "🆕 NEW ENTRY",
        momentum > 3 & total_mentions >= 10 ~ "🚀 EXPLOSIVE",
        momentum > 1.5 & days_active >= 3 ~ "📈 TRENDING UP",
        days_active >= 5 & avg_daily >= 10 ~ "🔥 SUSTAINED HEAT",
        momentum < 0.5 & days_since_first > 3 ~ "📉 COOLING OFF",
        TRUE ~ "👀 WATCHING"
      )
    ) %>%
    arrange(desc(trend_score))
  
  return(analysis)
}

# Generate report
generate_report <- function(analysis) {
  if(is.null(analysis) || nrow(analysis) == 0) return()
  
  cat("\n", rep("=", 60), "\n", sep="")
  cat("REDDIT SENTIMENT TREND ANALYSIS\n")
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
  cat(rep("=", 60), "\n\n", sep="")
  
  # Top movers by category
  categories <- c("🆕 NEW ENTRY", "🚀 EXPLOSIVE", "📈 TRENDING UP", 
                 "🔥 SUSTAINED HEAT", "👀 WATCHING")
  
  for(cat in categories) {
    stocks <- analysis %>% filter(category == cat)
    if(nrow(stocks) > 0) {
      cat(sprintf("\n%s (%d stocks)\n", cat, nrow(stocks)))
      cat(rep("-", 40), "\n", sep="")
      
      for(i in 1:min(5, nrow(stocks))) {
        s <- stocks[i,]
        cat(sprintf("  %-5s | Mentions: %3.0f | Momentum: %5.1fx | Days: %d | Score: %.1f\n",
                   s$ticker, s$total_mentions, s$momentum, s$days_active, s$trend_score))
        
        # Special alerts
        if(s$momentum > 5) {
          cat(sprintf("         ⚠️  ALERT: %d%% growth in mentions!\n", round(s$growth_pct)))
        }
        if(s$days_since_first == 0) {
          cat("         ⚡ First appearance today!\n")
        }
      }
    }
  }
  
  # Top 10 overall
  cat("\n\n📊 TOP 10 TRENDING STOCKS BY SCORE\n")
  cat(rep("=", 60), "\n", sep="")
  
  top10 <- head(analysis, 10)
  for(i in 1:nrow(top10)) {
    s <- top10[i,]
    cat(sprintf("%2d. %-5s %s\n", i, s$ticker, s$category))
    cat(sprintf("    Total: %3.0f mentions | Momentum: %.1fx | Active: %d days\n",
               s$total_mentions, s$momentum, s$days_active))
    cat(sprintf("    First seen: %s | Last seen: %s\n",
               s$first_seen, s$last_seen))
    if(s$growth_pct > 100) {
      cat(sprintf("    📈 %+.0f%% growth in recent activity\n", s$growth_pct))
    }
    cat("\n")
  }
  
  # Save report
  report_file <- sprintf("data/traction/trend_report_%s.txt", 
                        format(Sys.time(), "%Y%m%d_%H%M"))
  
  sink(report_file)
  cat("REDDIT TREND ANALYSIS - ", format(Sys.time(), "%Y-%m-%d %H:%M"), "\n\n")
  print(analysis %>% select(ticker, category, total_mentions, momentum, 
                            days_active, trend_score) %>% head(20))
  sink()
  
  cat("\n✅ Full report saved to:", report_file, "\n")
  
  # Save CSV for further analysis
  write_csv(analysis, "data/traction/trend_analysis_latest.csv")
  cat("✅ Data saved to: data/traction/trend_analysis_latest.csv\n")
  
  return(analysis)
}

# Plot trends for top stocks
plot_trends <- function(analysis, history_file = "data/traction/history/mention_history.csv") {
  if(!require(ggplot2)) {
    cat("Install ggplot2 for charts: install.packages('ggplot2')\n")
    return()
  }
  
  history <- read_csv(history_file, show_col_types = FALSE)
  
  # Get top 5 trending
  top_tickers <- head(analysis$ticker, 5)
  
  plot_data <- history %>%
    filter(ticker %in% top_tickers) %>%
    mutate(date = as.Date(timestamp)) %>%
    group_by(ticker, date) %>%
    summarise(mentions = sum(mentions), .groups = "drop")
  
  p <- ggplot(plot_data, aes(x = date, y = mentions, color = ticker)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    facet_wrap(~ticker, scales = "free_y") +
    theme_minimal() +
    labs(title = "Reddit Mention Trends - Top 5 Stocks",
         x = "Date", y = "Daily Mentions") +
    theme(legend.position = "none")
  
  ggsave("data/traction/trend_chart.png", p, width = 10, height = 6)
  cat("\n📊 Chart saved to: data/traction/trend_chart.png\n")
}

# Main execution
if(sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if(length(args) > 0 && args[1] == "scrape") {
    # Just run scraper
    run_scraper()
  } else if(length(args) > 0 && args[1] == "analyze") {
    # Just analyze
    analysis <- analyze_trends()
    if(!is.null(analysis)) {
      report <- generate_report(analysis)
      # plot_trends(analysis)  # Uncomment if ggplot2 installed
    }
  } else {
    # Do both
    run_scraper()
    cat("\n")
    analysis <- analyze_trends()
    if(!is.null(analysis)) {
      report <- generate_report(analysis)
    }
  }
}