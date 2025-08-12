# R Pipeline: Social Sentiment → Traction Watchlist
# -------------------------------------------------
# Scrapes Reddit for ticker mentions, scores sentiment + buzz, computes a
# "Hype Index", and outputs a daily watchlist of stocks likely to gain traction.
#
# Key Outputs:
# - data/traction/current_hype_watchlist.csv
# - data/traction/hype_features.parquet
# - data/traction/last_run_log.json
#
# Dependencies (install once):
# install.packages(c(
#   "RedditExtractoR","data.table","dplyr","stringr","lubridate","arrow",
#   "sentimentr","jsonlite","zoo","tibble","scales","readr"
# ))

suppressPackageStartupMessages({
  library(RedditExtractoR)
  library(data.table)
  library(dplyr)
  library(stringr)
  library(lubridate)
  library(arrow)
  library(sentimentr)
  library(jsonlite)
  library(zoo)
  library(tibble)
  library(scales)
  library(readr)
})

# -----------------------------
# 0) CONFIG
# -----------------------------
# Get script directory (works for source(), Rscript, and RStudio)
SCRIPT_DIR <- tryCatch({
  # When sourced with chdir = TRUE
  dirname(sys.frame(1)$ofile)
}, error = function(e) {
  # When run via Rscript
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if(length(file_arg) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_arg)))
  } else {
    # Fallback to working directory
    getwd()
  }
})

CONFIG <- list(
  run_id = format(Sys.time(), "%Y%m%d_%H%M%S"),
  storage_dir = file.path(SCRIPT_DIR, "data", "traction"),  # Save in script's folder
  lookback_hours = 24,            # Window for current signals
  baseline_days = 7,              # Baseline for growth comparisons
  scrape_subreddits = c("stocks","investing","wallstreetbets","StockMarket","pennystocks"),
  tickers = c("AAPL","AMZN","TSLA","NVDA","MSFT","AMD","META","GOOGL","ABX.TO","SHOP.TO"),
  ticker_regex_extra = c("\\$[A-Za-z]{1,5}", "[A-Z]{1,5}\\.TO"),
  score_high_threshold = 500,
  min_rows_for_zscore = 10,
  # Hype Index weights
  w_mentions_growth = 0.35,
  w_engagement_z     = 0,
  w_sentiment_avg    = 0.20,
  w_sentiment_momo   = 0.20,
  # Alert thresholds
  min_mentions_24h = 10,
  min_growth_pct   = 100,
  min_hype_index   = 1.2
)

FEATURES_PATH <- file.path(CONFIG$storage_dir, "hype_features.parquet")
WATCHLIST_CSV <- file.path(CONFIG$storage_dir, "current_hype_watchlist.csv")
LAST_LOG_JSON <- file.path(CONFIG$storage_dir, "last_run_log.json")
dir.create(CONFIG$storage_dir, recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 1) UTILITIES
# -----------------------------
log_msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%Y-%m-%d %H:%M:%S")), sprintf(...), "\n")

safe_str <- function(x) {
  if (is.null(x)) return("")
  x <- as.character(x)
  # Normalize encoding first; replace invalid bytes with a space
  x <- iconv(x, from = "", to = "UTF-8", sub = " ")
  # Collapse newlines/tabs and strip control chars using hex ranges (no NUL escapes)
  x <- gsub("[\\r\\n\\t]+", " ", x, perl = TRUE)
  x <- gsub("[\\x00-\\x1F\\x7F]+", " ", x, perl = TRUE)
  x <- gsub(" +", " ", x)
  trimws(x)
}


# Robust parser for Reddit time fields (POSIXct, epoch, ISO8601 with trailing Z)
safe_parse_reddit_time <- function(x) {
  if (inherits(x, "POSIXt")) return(lubridate::with_tz(x, "UTC"))
  if (is.numeric(x))        return(lubridate::as_datetime(x, tz = "UTC"))
  
  x_chr <- trimws(as.character(x))
  x_chr[x_chr == ""] <- NA_character_
  
  # Normalize ISO8601: replace 'T' with space, strip trailing 'Z'
  iso <- gsub("Z$", "", x_chr, ignore.case = TRUE)
  iso <- gsub("T", " ", iso, fixed = TRUE)
  
  out <- rep(as.POSIXct(NA, tz = "UTC"), length(iso))
  
  t1 <- suppressWarnings(lubridate::ymd_hms(iso, tz = "UTC", quiet = TRUE))
  out[is.na(out) & !is.na(t1)] <- t1[is.na(out) & !is.na(t1)]
  
  t2 <- suppressWarnings(lubridate::ymd_hm(iso, tz = "UTC", quiet = TRUE))
  out[is.na(out) & !is.na(t2)] <- t2[is.na(out) & !is.na(t2)]
  
  t3 <- suppressWarnings(lubridate::mdy_hms(x_chr, tz = "UTC", quiet = TRUE))
  out[is.na(out) & !is.na(t3)] <- t3[is.na(out) & !is.na(t3)]
  
  t4 <- suppressWarnings(lubridate::mdy_hm(x_chr, tz = "UTC", quiet = TRUE))
  out[is.na(out) & !is.na(t4)] <- t4[is.na(out) & !is.na(t4)]
  
  t5 <- suppressWarnings(lubridate::ymd(x_chr, tz = "UTC", quiet = TRUE))
  out[is.na(out) & !is.na(t5)] <- t5[is.na(out) & !is.na(t5)]
  
  suppressWarnings({ numlike <- as.numeric(x_chr) })
  idx_num <- !is.na(numlike) & grepl("^[0-9]+(\\.[0-9]+)?$", x_chr)
  if (any(idx_num)) out[idx_num & is.na(out)] <- lubridate::as_datetime(numlike[idx_num & is.na(out)], tz = "UTC")
  
  out
}

# Pick the first existing column from a data.frame
get_first_existing_col <- function(df, candidates) {
  nm <- candidates[candidates %in% names(df)]
  if (length(nm) == 0) return(rep(NA_character_, nrow(df)))
  df[[nm[1]]]
}

# Coalesce list elements by name (for tc$threads fields)
coalesce_value <- function(lst, candidates) {
  for (nm in candidates) if (!is.null(lst[[nm]])) return(lst[[nm]])
  NA
}

# Finance slang/emoji booster
SLANG <- tibble::tibble(
  pattern = c("🚀|moon|to the moon|tendies|ATH|rip(p)?ing|squeeze|YOLO|diamond hands|breakout|gap up|parabolic",
              "baghold(er)?|dump(ed)?|rug|scam|dilution|halt(ed)?|bankrupt|bear trap|selloff|downtrend|gap down|rekt"),
  weight  = c(+0.6, -0.6)
)

apply_slang_boost <- function(text, base){
  boosts <- sapply(seq_len(nrow(SLANG)), function(i){
    ifelse(str_detect(tolower(text), SLANG$pattern[i]), SLANG$weight[i], 0)
  })
  base + sum(boosts)
}

# Extract tickers from free text
extract_tickers <- function(text, universe){
  pats <- c(CONFIG$ticker_regex_extra, paste0("\\b", universe, "\\b"))
  rx <- paste(pats, collapse = "|")
  unique(stringr::str_extract_all(text, rx, simplify = FALSE)[[1]])
}

# Robust z-score with fallback
rz <- function(x){
  x <- as.numeric(x)
  if(length(na.omit(x)) < CONFIG$min_rows_for_zscore) return(rep(0, length(x)))
  m <- median(x, na.rm = TRUE)
  s <- mad(x, constant = 1.4826, na.rm = TRUE)
  if(is.na(s) || s == 0) s <- sd(x, na.rm = TRUE)
  z <- ifelse(s > 0, (x - m)/s, 0)
  replace(z, is.na(z), 0)
}

# -----------------------------
# 2) SCRAPE REDDIT (threads + comments)
# -----------------------------
# - find_thread_urls(subreddit = "stocks", sort_by = "new", period = "day")
# - get_thread_content(url)

scrape_reddit_batch <- function(subs, lookback_hours){
  end_time <- Sys.time()
  start_time <- end_time - hours(lookback_hours)
  
  all_posts <- list()
  
  for(sb in subs){
    log_msg("Fetching thread URLs from r/%s", sb)
    urls <- tryCatch({
      find_thread_urls(subreddit = sb, sort_by = "new", period = "day")
    }, error = function(e){
      log_msg("WARN: find_thread_urls failed for %s: %s", sb, e$message)
      NULL
    })
    
    if (is.null(urls) || nrow(urls) == 0) next
    
    # Time filter (robust parsing)
    urls <- urls %>% mutate(
      date_raw = get_first_existing_col(cur_data_all(), c("date_utc","date","timestamp")),
      date     = safe_parse_reddit_time(date_raw)
    ) %>% filter(!is.na(date) & date >= with_tz(start_time, "UTC"))
    
    if(nrow(urls) == 0){
      log_msg("INFO: No URLs within window after parsing times for r/%s.", sb)
      next
    }
    
    # Ensure subreddit column exists
    if (!"subreddit" %in% names(urls)) urls$subreddit <- sb
    
    for(u in urls$url){
      Sys.sleep(0.7) # be nice
      tc <- tryCatch(get_thread_content(u), error = function(e) NULL)
      if (is.null(tc)) next
      
      post_dt <- safe_parse_reddit_time(coalesce_value(tc$threads, c("date_utc","date","timestamp")))
      if(length(post_dt) == 0 || is.na(post_dt)) next
      if(post_dt < with_tz(start_time, "UTC")) next
      
      post_row <- tibble(
        type = "post",
        url = u,
        subreddit = tc$threads$subreddit %||% sb,
        author = tc$threads$author,
        score = suppressWarnings(as.numeric(tc$threads$score)),
        num_comments = suppressWarnings(as.numeric(tc$threads$comments)),
        text = safe_str(paste(tc$threads$title, tc$threads$text, collapse = " ")),
        created_utc = post_dt
      )
      
      comm <- NULL
      if(!is.null(tc$comments) && nrow(tc$comments) > 0){
        comm_src <- tc$comments
        author_col <- get_first_existing_col(comm_src, c("author","comment_author","user"))
        score_col  <- get_first_existing_col(comm_src, c("comment_score","score","ups","upvotes"))
        text_col   <- get_first_existing_col(comm_src, c("comment","body","text"))
        date_col   <- get_first_existing_col(comm_src, c("date_utc","date","timestamp"))
        
        comm <- tibble(
          type = "comment",
          url  = u,
          subreddit = tc$threads$subreddit %||% sb,
          author = author_col,
          score  = suppressWarnings(as.numeric(score_col)),
          num_comments = NA_real_,
          text = safe_str(text_col),
          created_utc = safe_parse_reddit_time(date_col)
        ) %>% filter(!is.na(created_utc) & created_utc >= with_tz(start_time, "UTC"))
      }
      
      all_posts[[length(all_posts)+1]] <- bind_rows(post_row, comm)
    }
  }
  
  if(length(all_posts) == 0) return(tibble())
  bind_rows(all_posts) %>% distinct(url, type, text, created_utc, .keep_all = TRUE)
}

# -----------------------------
# 3) SENTIMENT
# -----------------------------
score_sentiment <- function(vec_text){
  if(length(vec_text) == 0) return(numeric(0))
  base <- tryCatch({ sentiment_by(vec_text)$ave_sentiment }, error = function(e){ rep(0, length(vec_text)) })
  mapply(apply_slang_boost, vec_text, base)
}

# -----------------------------
# 4) PIPE: CLEAN → TICKERS → FEATURES
# -----------------------------
process_batch <- function(raw_dt, universe){
  if(nrow(raw_dt) == 0) return(tibble())
  
  dt <- raw_dt %>% mutate(
    text = safe_str(text),
    engagement = pmax( ifelse(is.na(score), 0, score) + ifelse(is.na(num_comments), 0, num_comments), 0),
    influencer = ifelse(!is.na(score) & score >= CONFIG$score_high_threshold, 1, 0),
    sent = score_sentiment(text)
  )
  
  rows <- lapply(seq_len(nrow(dt)), function(i){
    tix <- extract_tickers(dt$text[i], universe)
    if(length(tix) == 0) return(NULL)
    tibble(
      created_utc = dt$created_utc[i],
      subreddit = dt$subreddit[i],
      url = dt$url[i],
      type = dt$type[i],
      author = dt$author[i],
      score = dt$score[i],
      num_comments = dt$num_comments[i],
      engagement = dt$engagement[i],
      influencer = dt$influencer[i],
      sent = dt$sent[i],
      ticker_raw = tix
    )
  })
  
  exploded <- bind_rows(rows)
  if(nrow(exploded) == 0) return(tibble())
  
  exploded <- exploded %>% mutate(
    ticker = str_replace_all(ticker_raw, "^\\$", ""),
    ticker = str_replace_all(ticker, "\\$", ""),
    ticker = toupper(ticker)
  )
  
  exploded %>%
    mutate(hour_utc = floor_date(created_utc, unit = "hour")) %>%
    group_by(ticker, hour_utc) %>%
    summarise(
      mentions = n(),
      sent_avg = mean(sent, na.rm = TRUE),
      sent_med = median(sent, na.rm = TRUE),
      engagement_sum = sum(engagement, na.rm = TRUE),
      influencers = sum(influencer, na.rm = TRUE),
      .groups = "drop"
    )
}

# -----------------------------
# 5) FEATURE STORE & HYPE INDEX
# -----------------------------
# Atomic Parquet writer with retry (helps with OneDrive/Windows locks)
safe_write_parquet <- function(df, path, retries = 5, sleep_sec = 0.5) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  for (i in seq_len(retries)) {
    tmp <- paste0(path, ".tmp-", as.integer(runif(1, 1, 1e9)))
    ok <- FALSE
    try({
      arrow::write_parquet(df, tmp)
      ok <- file.rename(tmp, path)  # atomic replace
    }, silent = TRUE)
    if (ok) return(invisible(TRUE))
    if (file.exists(tmp)) unlink(tmp)
    Sys.sleep(sleep_sec * i)  # backoff
  }
  stop(sprintf("Failed to write parquet after %d retries: %s", retries, path))
}

update_feature_store <- function(hourly){
  if(nrow(hourly) == 0) return(hourly)
  if(file.exists(FEATURES_PATH)){
    existing <- as.data.table(read_parquet(FEATURES_PATH))
    combined <- rbindlist(list(existing, as.data.table(hourly)), use.names = TRUE, fill = TRUE)
    combined <- unique(combined, by = c("ticker","hour_utc"))
  } else {
    combined <- as.data.table(hourly)
  }
  setorder(combined, ticker, hour_utc)
  safe_write_parquet(combined, FEATURES_PATH)
  combined
}

compute_hype_now <- function(features_dt){
  if(nrow(features_dt) == 0) return(tibble())
  
  now_end   <- floor_date(with_tz(Sys.time(), "UTC"), unit = "hour")
  now_start <- now_end - hours(CONFIG$lookback_hours - 1)
  base_start <- now_start - days(CONFIG$baseline_days)
  
  dt <- features_dt %>% filter(hour_utc >= base_start & hour_utc <= now_end)
  if(nrow(dt) == 0) return(tibble())
  
  current  <- dt %>% filter(hour_utc >= now_start)
  baseline <- dt %>% filter(hour_utc <  now_start)
  
  # Basic 24h aggregates per ticker
  cur_basic <- current %>% group_by(ticker) %>% summarise(
    mentions_24h    = sum(mentions, na.rm = TRUE),
    sent_avg_24h    = weighted.mean(sent_avg, w = pmax(1, mentions), na.rm = TRUE),
    engagement_24h  = sum(engagement_sum, na.rm = TRUE),
    influencers_24h = sum(influencers, na.rm = TRUE),
    .groups = "drop"
  )
  
  # Sentiment momentum (robust to sparse hours): mean(last 6h) - mean(prev 6h)
  sent_momo_tbl <- current %>%
    arrange(ticker, hour_utc) %>%
    group_by(ticker) %>%
    summarise(
      n_last6  = sum(hour_utc >= now_end - hours(5)),
      n_prev6  = sum(hour_utc >= now_end - hours(11) & hour_utc < now_end - hours(5)),
      sent_last6 = mean(sent_avg[hour_utc >= now_end - hours(5)], na.rm = TRUE),
      sent_prev6 = mean(sent_avg[hour_utc >= now_end - hours(11) & hour_utc < now_end - hours(5)], na.rm = TRUE),
      sent_momo  = ifelse(n_last6 >= 2 & n_prev6 >= 2, dplyr::coalesce(sent_last6 - sent_prev6, 0), 0),
      .groups = "drop"
    )
  
  cur_agg <- cur_basic %>% left_join(sent_momo_tbl, by = "ticker")
  
  base_agg <- baseline %>% group_by(ticker) %>% summarise(
    mentions_baseline   = mean(mentions, na.rm = TRUE) * 24,
    engagement_baseline = mean(engagement_sum, na.rm = TRUE) * 24,
    .groups = "drop"
  )
  
  out <- cur_agg %>% left_join(base_agg, by = "ticker") %>% mutate(
    mentions_baseline   = ifelse(is.na(mentions_baseline), 0, mentions_baseline),
    engagement_baseline = ifelse(is.na(engagement_baseline), 0, engagement_baseline),
    # set reasonable floors to avoid divide-by-almost-zero explosions
    mentions_base_floor   = pmax(mentions_baseline,   5),
    engagement_base_floor = pmax(engagement_baseline, 50),
    mentions_growth_pct   = 100 * (mentions_24h - mentions_baseline)   / mentions_base_floor,
    engagement_growth_pct = 100 * (engagement_24h - engagement_baseline) / engagement_base_floor
  )
  
  dz <- dt %>% group_by(ticker) %>% summarise(
    engagement_z = tail(rz(engagement_sum), 1),
    mentions_z   = tail(rz(mentions), 1),
    .groups = "drop"
  )
  
  out <- out %>% left_join(dz, by = "ticker") %>% mutate(
    hype_index = CONFIG$w_mentions_growth * scales::rescale(mentions_growth_pct, to = c(0, 3), from = range(mentions_growth_pct, na.rm = TRUE)) +
      CONFIG$w_engagement_z     * engagement_z +
      CONFIG$w_sentiment_avg    * sent_avg_24h +
      CONFIG$w_sentiment_momo   * sent_momo
  ) %>% arrange(desc(hype_index))
  
  out %>% mutate(
    alert = (mentions_24h >= CONFIG$min_mentions_24h) &
      (mentions_growth_pct >= CONFIG$min_growth_pct) &
      (hype_index >= CONFIG$min_hype_index)
  )
  
  out <- out %>%
    mutate(
      min_hours_covered = current %>% filter(ticker == .data$ticker) %>% n_distinct(hour_utc)
    )
}

# -----------------------------
# 6) MAIN RUNNER
# -----------------------------
run_once <- function(){
  t0 <- Sys.time()
  raw <- scrape_reddit_batch(CONFIG$scrape_subreddits, CONFIG$lookback_hours)
  log_msg("Scraped rows: %s", nrow(raw))
  
  hourly <- process_batch(raw, CONFIG$tickers)
  log_msg("Hourly rows: %s", nrow(hourly))
  
  features <- update_feature_store(hourly)
  hype <- compute_hype_now(features)
  
  if(nrow(hype) > 0){
    out <- hype %>% transmute(
      ticker,
      mentions_24h,
      mentions_growth_pct = round(mentions_growth_pct,1),
      sent_avg_24h = round(sent_avg_24h,3),
      sent_momo = round(sent_momo,3),
      engagement_24h,
      influencers_24h,
      hype_index = round(hype_index,3),
      alert
    )
    readr::write_csv(out, WATCHLIST_CSV)
  }
  
  log <- list(
    run_id = CONFIG$run_id,
    started_at = as.character(t0),
    ended_at = as.character(Sys.time()),
    scraped_rows = nrow(raw),
    hourly_rows = nrow(hourly),
    universe = CONFIG$tickers,
    subs = CONFIG$scrape_subreddits,
    lookback_hours = CONFIG$lookback_hours
  )
  write(jsonlite::toJSON(log, auto_unbox = TRUE, pretty = TRUE), LAST_LOG_JSON)
  
  invisible(list(hype = hype, features = features, hourly = hourly, raw = raw))
}

# -----------------------------
# 7) OPTIONAL: CRON
# -----------------------------
# */30 * * * * Rscript /path/to/this_script.R >> /path/to/data/traction/cron.log 2>&1

# -----------------------------
# 8) EXECUTE INTERACTIVELY
# -----------------------------
`%||%` <- function(a, b) if (!is.null(a)) a else b
if (sys.nframe() == 0) {
  res <- run_once()
  try(print(head(res$hype, 10)), silent = TRUE)
}
