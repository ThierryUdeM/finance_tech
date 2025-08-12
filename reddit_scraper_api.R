#!/usr/bin/env Rscript

# Reddit Scraper using direct API (avoids 403 errors)
# Works in GitHub Actions environment

suppressPackageStartupMessages({
  library(httr)
  library(jsonlite)
  library(dplyr)
  library(stringr)
  library(readr)
})

CONFIG <- list(
  storage_dir = "data/traction",
  nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
  subreddits = c("wallstreetbets", "stocks", "investing"),
  max_posts = 25,
  user_agent = "linux:reddit-sentiment:v1.0 (by /u/sentiment_bot)"
)

dir.create(CONFIG$storage_dir, recursive = TRUE, showWarnings = FALSE)

# Common words to exclude
EXCLUDE_COMMON <- c("ON", "CAN", "HAS", "ANY", "GOOD", "WAY", "GO", "APP", 
                   "EVER", "LOT", "OPEN", "REAL", "TOP", "ADD", "BEAT", 
                   "CASH", "FINE", "NOW", "ALL", "BE", "BY", "DO", "FOR",
                   "GET", "SEE", "ARE", "AT", "BIG", "BUT", "NEW", "ONE",
                   "OUT", "SO", "UP", "VERY", "WELL", "WILL", "WITH", "OR")

# Get Reddit data using direct API
get_reddit_posts <- function(subreddit, sort = "hot", limit = 25) {
  url <- sprintf("https://www.reddit.com/r/%s/%s.json?limit=%d", 
                 subreddit, sort, limit)
  
  response <- tryCatch({
    GET(url, 
        add_headers(`User-Agent` = CONFIG$user_agent),
        timeout(10))
  }, error = function(e) {
    cat(sprintf("  Error fetching %s: %s\n", subreddit, e$message))
    return(NULL)
  })
  
  if(is.null(response)) return(NULL)
  
  if(status_code(response) != 200) {
    cat(sprintf("  HTTP %d for r/%s\n", status_code(response), subreddit))
    return(NULL)
  }
  
  # Parse JSON
  content <- content(response, as = "text", encoding = "UTF-8")
  data <- fromJSON(content, flatten = TRUE)
  
  if(!is.null(data$data$children)) {
    posts <- data$data$children$data
    return(posts)
  }
  
  return(NULL)
}

# Get valid tickers
get_tickers <- function() {
  cache <- file.path(CONFIG$storage_dir, "valid_tickers.rds")
  if(file.exists(cache)) {
    tickers <- readRDS(cache)
    return(setdiff(tickers, EXCLUDE_COMMON))
  }
  
  cat("Downloading NASDAQ tickers...\n")
  tryCatch({
    data <- read_delim(CONFIG$nasdaq_url, delim = "|", show_col_types = FALSE)
    tickers <- data$Symbol[!grepl("File Creation", data$Symbol)]
    tickers <- tickers[!is.na(tickers)]
    tickers <- setdiff(tickers, EXCLUDE_COMMON)
    saveRDS(tickers, cache)
    tickers
  }, error = function(e) {
    # Fallback list
    c("AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "AMD", 
      "GME", "AMC", "SPY", "QQQ", "PLTR", "SOFI", "BB", "NOK")
  })
}

# Extract tickers from text
find_tickers_strict <- function(text, valid_list) {
  if(is.null(text) || text == "") return(character(0))
  
  text <- iconv(text, from = "", to = "UTF-8", sub = " ")
  text <- gsub("[^[:print:]]", " ", text)
  
  found <- c()
  text_upper <- toupper(text)
  
  # Cashtags ($AAPL)
  cashtags <- str_extract_all(text, "\\$[A-Z]{1,5}\\b")[[1]]
  if(length(cashtags) > 0) {
    cashtags <- gsub("\\$", "", toupper(cashtags))
    found <- c(found, cashtags[cashtags %in% valid_list])
  }
  
  # Parentheses (AAPL) and brackets [MSFT]
  parens <- str_extract_all(text_upper, "\\([A-Z]{2,5}\\)")[[1]]
  if(length(parens) > 0) {
    parens <- gsub("[\\(\\)]", "", parens)
    found <- c(found, parens[parens %in% valid_list])
  }
  
  brackets <- str_extract_all(text_upper, "\\[[A-Z]{2,5}\\]")[[1]]
  if(length(brackets) > 0) {
    brackets <- gsub("[\\[\\]]", "", brackets)
    found <- c(found, brackets[brackets %in% valid_list])
  }
  
  # Trading context patterns
  trading_patterns <- c(
    "(?:buy|buying|bought|sell|selling|sold|long|short)\\s+([A-Z]{2,5})\\b",
    "([A-Z]{2,5})\\s+(?:calls?|puts?|shares?|stock|position)",
    "([A-Z]{2,5})\\s+(?:moon|mooning|squeeze|earnings)"
  )
  
  for(pattern in trading_patterns) {
    matches <- str_extract_all(text_upper, regex(pattern))[[1]]
    if(length(matches) > 0) {
      tickers <- str_extract(matches, "\\b[A-Z]{2,5}\\b")
      found <- c(found, tickers[tickers %in% valid_list])
    }
  }
  
  unique(found)
}

# Main execution
main <- function() {
  cat("\n==== REDDIT TICKER SCANNER (API VERSION) ====\n\n")
  
  valid_tickers <- get_tickers()
  cat(sprintf("Using %d valid tickers\n\n", length(valid_tickers)))
  
  all_mentions <- list()
  
  for(sub in CONFIG$subreddits) {
    cat(sprintf("Fetching r/%s...\n", sub))
    
    # Try different sort methods
    posts <- NULL
    for(sort_method in c("new", "hot", "top")) {
      posts <- get_reddit_posts(sub, sort = sort_method, limit = CONFIG$max_posts)
      if(!is.null(posts) && nrow(posts) > 0) {
        cat(sprintf("  Got %d posts using sort=%s\n", nrow(posts), sort_method))
        break
      }
    }
    
    if(is.null(posts) || nrow(posts) == 0) {
      cat(sprintf("  No posts retrieved from r/%s\n", sub))
      next
    }
    
    # Process posts
    posts_with_tickers <- 0
    
    for(i in 1:nrow(posts)) {
      # Combine title and selftext
      title <- posts$title[i] %||% ""
      text <- posts$selftext[i] %||% ""
      full_text <- paste(title, text, sep = " ")
      
      tickers <- find_tickers_strict(full_text, valid_tickers)
      
      if(length(tickers) > 0) {
        posts_with_tickers <- posts_with_tickers + 1
        
        for(ticker in tickers) {
          all_mentions[[length(all_mentions) + 1]] <- data.frame(
            ticker = ticker,
            subreddit = sub,
            score = posts$score[i] %||% 0,
            num_comments = posts$num_comments[i] %||% 0,
            title = substr(title, 1, 100),
            stringsAsFactors = FALSE
          )
        }
      }
    }
    
    cat(sprintf("  Found tickers in %d posts\n", posts_with_tickers))
  }
  
  if(length(all_mentions) > 0) {
    df <- bind_rows(all_mentions)
    
    summary <- df %>%
      group_by(ticker) %>%
      summarise(
        mentions = n(),
        avg_score = mean(score, na.rm = TRUE),
        total_comments = sum(num_comments, na.rm = TRUE),
        subreddits = n_distinct(subreddit),
        .groups = "drop"
      ) %>%
      arrange(desc(mentions))
    
    cat("\n==== TOP MENTIONED TICKERS ====\n\n")
    for(i in 1:min(20, nrow(summary))) {
      cat(sprintf("%2d. %-5s : %2d mentions (avg score: %.0f, %d subs)\n", 
                  i, 
                  summary$ticker[i], 
                  summary$mentions[i],
                  summary$avg_score[i],
                  summary$subreddits[i]))
    }
    
    # Save results
    write_csv(summary, file.path(CONFIG$storage_dir, "reddit_mentions_strict.csv"))
    cat(sprintf("\n✅ Saved %d tickers to: data/traction/reddit_mentions_strict.csv\n", nrow(summary)))
    
  } else {
    # Create empty file with headers
    empty_df <- data.frame(
      ticker = character(),
      mentions = integer(),
      avg_score = numeric(),
      total_comments = integer(),
      subreddits = integer()
    )
    write_csv(empty_df, file.path(CONFIG$storage_dir, "reddit_mentions_strict.csv"))
    
    cat("\n❌ No tickers found\n")
    cat("   Possible reasons:\n")
    cat("   - Reddit API rate limiting\n")
    cat("   - No explicit ticker mentions in recent posts\n")
    cat("   - Try again in a few minutes\n")
  }
}

# Helper
`%||%` <- function(a, b) if(!is.null(a) && length(a) > 0) a else b

# Run
if(sys.nframe() == 0) {
  main()
}