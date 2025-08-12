#!/usr/bin/env Rscript

# Reddit Sentiment Scanner - STRICT VERSION
# Only finds tickers with explicit mentions ($TICKER, (TICKER), etc.)

suppressPackageStartupMessages({
  library(RedditExtractoR)
  library(dplyr)
  library(stringr) 
  library(readr)
})

CONFIG <- list(
  storage_dir = "data/traction",
  nasdaq_url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
  subreddits = c("wallstreetbets", "stocks", "investing"),
  max_posts = 15
)

dir.create(CONFIG$storage_dir, recursive = TRUE, showWarnings = FALSE)

# Common words that are also tickers to exclude
EXCLUDE_COMMON <- c("ON", "CAN", "HAS", "ANY", "GOOD", "WAY", "GO", "APP", 
                   "EVER", "LOT", "OPEN", "REAL", "TOP", "ADD", "BEAT", 
                   "CASH", "FINE", "NOW", "ALL", "BE", "BY", "DO", "FOR",
                   "GET", "SEE", "ARE", "AT", "BIG", "BUT", "NEW", "ONE",
                   "OUT", "SO", "UP", "VERY", "WELL", "WILL", "WITH", "OR",
                   "ELSE", "NEXT", "OLD", "TURN", "AWAY", "GLAD", "NICE")

# Get ticker list
get_tickers <- function() {
  cache <- file.path(CONFIG$storage_dir, "valid_tickers.rds")
  if(file.exists(cache)) {
    tickers <- readRDS(cache)
    # Remove common words
    return(setdiff(tickers, EXCLUDE_COMMON))
  }
  
  cat("Downloading NASDAQ tickers...\n")
  tryCatch({
    data <- read_delim(CONFIG$nasdaq_url, delim = "|", show_col_types = FALSE)
    tickers <- data$Symbol[!grepl("File Creation", data$Symbol)]
    tickers <- tickers[!is.na(tickers)]
    # Remove common words
    tickers <- setdiff(tickers, EXCLUDE_COMMON)
    saveRDS(tickers, cache)
    tickers
  }, error = function(e) {
    c("AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN", "AMD", 
      "GME", "AMC", "SPY", "QQQ", "PLTR", "SOFI")
  })
}

# STRICT ticker extraction
find_tickers_strict <- function(text, valid_list) {
  if(is.null(text) || text == "") return(character(0))
  
  # Clean text
  text <- iconv(text, from = "", to = "UTF-8", sub = " ")
  text <- gsub("[^[:print:]]", " ", text)
  
  found <- c()
  text_upper <- toupper(text)
  
  # 1. EXPLICIT CASHTAGS - Most reliable ($AAPL, $TSLA)
  cashtags <- str_extract_all(text, "\\$[A-Z]{1,5}\\b")[[1]]
  if(length(cashtags) > 0) {
    cashtags <- gsub("\\$", "", toupper(cashtags))
    cashtags <- cashtags[cashtags %in% valid_list]
    found <- c(found, cashtags)
  }
  
  # 2. TICKERS IN PARENTHESES - Often used for clarity (AAPL), [MSFT]
  parens <- str_extract_all(text_upper, "\\([A-Z]{2,5}\\)")[[1]]
  if(length(parens) > 0) {
    parens <- gsub("[\\(\\)]", "", parens)
    parens <- parens[parens %in% valid_list]
    found <- c(found, parens)
  }
  
  brackets <- str_extract_all(text_upper, "\\[[A-Z]{2,5}\\]")[[1]]
  if(length(brackets) > 0) {
    brackets <- gsub("[\\[\\]]", "", brackets)
    brackets <- brackets[brackets %in% valid_list]
    found <- c(found, brackets)
  }
  
  # 3. EXPLICIT MENTIONS with trading keywords
  # "buy AAPL", "TSLA calls", "sold MSFT", etc.
  trading_patterns <- c(
    "(?:buy|buying|bought|grab|grabbed|loading|loaded)\\s+([A-Z]{2,5})\\b",
    "(?:sell|selling|sold|dump|dumped|dumping|short|shorting)\\s+([A-Z]{2,5})\\b",
    "([A-Z]{2,5})\\s+(?:calls?|puts?|options?|shares?|stock|position)",
    "(?:long|bullish|bearish)\\s+(?:on\\s+)?([A-Z]{2,5})\\b",
    "([A-Z]{2,5})\\s+(?:moon|mooning|squeeze|squeezing|earnings|dividend)",
    "\\bDD\\s+(?:on\\s+)?([A-Z]{2,5})\\b",  # DD on TICKER
    "([A-Z]{2,5})\\s+(?:to\\s+)?(?:the\\s+)?moon",  # TICKER to the moon
    "([A-Z]{2,5})\\s+\\d+[cp]\\b"  # TICKER 420c, TICKER 69p
  )
  
  for(pattern in trading_patterns) {
    matches <- str_extract_all(text_upper, regex(pattern))[[1]]
    if(length(matches) > 0) {
      # Extract just the ticker part
      tickers <- str_extract(matches, "\\b[A-Z]{2,5}\\b")
      tickers <- tickers[tickers %in% valid_list]
      found <- c(found, tickers)
    }
  }
  
  # 4. Tickers mentioned with : or =
  # "ticker: AAPL", "symbol = MSFT"
  explicit_patterns <- c(
    "(?:ticker|symbol|stock|company)\\s*[:=]\\s*([A-Z]{2,5})\\b",
    "\\$([A-Z]{2,5})\\b"  # Cashtag without the $
  )
  
  for(pattern in explicit_patterns) {
    matches <- str_extract_all(text_upper, regex(pattern))[[1]]
    if(length(matches) > 0) {
      tickers <- str_extract(matches, "[A-Z]{2,5}")
      tickers <- tickers[tickers %in% valid_list]
      found <- c(found, tickers)
    }
  }
  
  # 5. Popular tickers in clear financial context
  # Only allow well-known tickers if mentioned with price or %
  if(any(str_detect(text, c("\\$\\d+", "\\d+%", "\\d+\\s*percent")))) {
    popular <- c("AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", 
                 "AMD", "INTC", "SPY", "QQQ", "GME", "AMC", "PLTR")
    
    for(ticker in popular) {
      if(ticker %in% valid_list && str_detect(text_upper, paste0("\\b", ticker, "\\b"))) {
        # Check it's near a price or percentage
        pattern <- paste0(ticker, ".{0,20}(\\$\\d+|\\d+%)|\\$\\d+.{0,20}", ticker, "|\\d+%.{0,20}", ticker)
        if(str_detect(text_upper, pattern)) {
          found <- c(found, ticker)
        }
      }
    }
  }
  
  unique(found)
}

# Helper
`%||%` <- function(a, b) if(!is.null(a) && length(a) > 0) a else b

# Main
cat("\n==== REDDIT TICKER SCANNER (STRICT) ====\n\n")

valid_tickers <- get_tickers()
cat(sprintf("Using %d valid tickers (excluding common words)\n\n", length(valid_tickers)))

all_mentions <- list()

for(sub in CONFIG$subreddits) {
  cat(sprintf("Fetching r/%s...\n", sub))
  
  # Get URLs
  urls <- tryCatch({
    find_thread_urls(subreddit = sub, sort_by = "hot")
  }, error = function(e) {
    cat(sprintf("  Error: %s\n", e$message))
    NULL
  })
  
  if(is.null(urls) || nrow(urls) == 0) next
  
  urls <- head(urls, CONFIG$max_posts)
  cat(sprintf("  Processing %d posts...\n", nrow(urls)))
  
  posts_with_tickers <- 0
  
  for(i in 1:nrow(urls)) {
    # Get content
    content <- tryCatch({
      get_thread_content(urls$url[i])
    }, error = function(e) NULL)
    
    if(is.null(content)) next
    
    found_in_post <- FALSE
    
    # Check title and text
    if(!is.null(content$threads)) {
      title <- content$threads$title %||% ""
      text <- content$threads$text %||% ""
      
      # Clean encoding
      title <- iconv(title, from = "", to = "UTF-8", sub = " ")
      text <- iconv(text, from = "", to = "UTF-8", sub = " ")
      
      full_text <- paste(title, text, collapse = " ")
      
      tickers <- find_tickers_strict(full_text, valid_tickers)
      
      if(length(tickers) > 0) {
        found_in_post <- TRUE
        for(ticker in tickers) {
          all_mentions[[length(all_mentions) + 1]] <- data.frame(
            ticker = ticker,
            subreddit = sub,
            type = "post",
            title = substr(title, 1, 100),
            stringsAsFactors = FALSE
          )
        }
      }
    }
    
    # Check comments (only first 15)
    if(!is.null(content$comments) && nrow(content$comments) > 0) {
      for(j in 1:min(15, nrow(content$comments))) {
        comment_text <- content$comments$comment[j] %||% 
                       content$comments$text[j] %||% 
                       content$comments$body[j] %||% ""
        
        if(nchar(comment_text) > 5) {
          tickers <- find_tickers_strict(comment_text, valid_tickers)
          
          if(length(tickers) > 0) {
            found_in_post <- TRUE
            for(ticker in tickers) {
              all_mentions[[length(all_mentions) + 1]] <- data.frame(
                ticker = ticker,
                subreddit = sub,
                type = "comment",
                title = "",
                stringsAsFactors = FALSE
              )
            }
          }
        }
      }
    }
    
    if(found_in_post) posts_with_tickers <- posts_with_tickers + 1
    Sys.sleep(0.3)  # Rate limit
  }
  
  cat(sprintf("  Found tickers in %d posts\n", posts_with_tickers))
}

if(length(all_mentions) > 0) {
  df <- bind_rows(all_mentions)
  
  # Summary
  summary <- df %>%
    group_by(ticker) %>%
    summarise(
      mentions = n(),
      posts = sum(type == "post"),
      comments = sum(type == "comment"),
      subreddits = n_distinct(subreddit),
      .groups = "drop"
    ) %>%
    arrange(desc(mentions))
  
  # Show sample contexts
  cat("\n==== TOP MENTIONED TICKERS (STRICT) ====\n\n")
  for(i in 1:min(20, nrow(summary))) {
    cat(sprintf("%2d. %-5s : %2d mentions (%d posts, %d comments) in %d sub(s)\n", 
                i, 
                summary$ticker[i], 
                summary$mentions[i],
                summary$posts[i],
                summary$comments[i],
                summary$subreddits[i]))
    
    # Show a sample title for context
    if(summary$posts[i] > 0) {
      sample_title <- df %>% 
        filter(ticker == summary$ticker[i], type == "post", title != "") %>%
        pull(title) %>%
        first()
      
      if(!is.na(sample_title) && nchar(sample_title) > 0) {
        cat(sprintf("     Context: \"%s...\"\n", substr(sample_title, 1, 60)))
      }
    }
  }
  
  # Save
  write_csv(summary, file.path(CONFIG$storage_dir, "reddit_mentions_strict.csv"))
  cat(sprintf("\n✅ Saved %d unique tickers to: data/traction/reddit_mentions_strict.csv\n", nrow(summary)))
  cat("✅ Only found tickers with explicit mentions (cashtags, parentheses, trading context)\n")
  cat("✅ Excluded common words that happen to be tickers\n")
  
} else {
  cat("\n❌ No tickers found with strict criteria\n")
  cat("   This could mean:\n")
  cat("   - Reddit posts don't have explicit ticker mentions\n")
  cat("   - Posts are using ticker names without $ or ()\n")
  cat("   - Try checking wallstreetbets directly for more trading-focused content\n")
}