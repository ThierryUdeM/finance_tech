library(gmailr)
library(dplyr)
library(stringr)
library(base64enc)
library(tidyr)
library(purrr)


# Try to load your client config directly with gargle
client <- gargle::oauth_app_from_json("client_secret_948428206366-q9aptd5ffh3c95c5igoq6l5ac14ee3ll.apps.googleusercontent.com.json")

print(client)

# Configure gmailr with the client object you loaded
gmailr::gm_auth_configure(client = client)


msgs <- gmailr::gm_messages(search = 'subject:"your order has been filled "')
print(msgs)

# Function to safely extract email content
extract_email_content <- function(msg) {
  tryCatch({
    # Get the full message
    full_msg <- gm_message(msg$id)
    
    # Extract basic info
    headers <- full_msg$payload$headers
    
    # Helper function to get header value
    get_header <- function(headers, name) {
      header <- headers[sapply(headers, function(x) x$name == name)]
      if (length(header) > 0) {
        return(header[[1]]$value)
      }
      return(NA)
    }
    
    # Extract headers
    from <- get_header(headers, "From")
    to <- get_header(headers, "To")
    subject <- get_header(headers, "Subject")
    date <- get_header(headers, "Date")
    
    # Extract body
    body <- extract_body(full_msg$payload)
    
    # Return as a list
    return(list(
      message_id = msg$id,
      thread_id = msg$threadId,
      from = from,
      to = to,
      subject = subject,
      date = date,
      body = body,
      snippet = full_msg$snippet
    ))
    
  }, error = function(e) {
    # Return NA values if there's an error
    return(list(
      message_id = msg$id,
      thread_id = msg$threadId,
      from = NA,
      to = NA,
      subject = NA,
      date = NA,
      body = paste("Error extracting message:", e$message),
      snippet = NA
    ))
  })
}

# Function to extract body from message payload
extract_body <- function(payload) {
  body <- ""
  
  # Check if payload has parts
  if (!is.null(payload$parts)) {
    for (part in payload$parts) {
      if (part$mimeType == "text/plain") {
        if (!is.null(part$body$data)) {
          decoded <- base64decode(gsub("-", "+", gsub("_", "/", part$body$data)))
          body <- paste(body, rawToChar(decoded), sep = "\n")
        }
      } else if (part$mimeType == "text/html" && body == "") {
        # Use HTML if plain text not available
        if (!is.null(part$body$data)) {
          decoded <- base64decode(gsub("-", "+", gsub("_", "/", part$body$data)))
          body <- rawToChar(decoded)
          # Basic HTML tag removal
          body <- gsub("<[^>]+>", "", body)
          body <- gsub("&nbsp;", " ", body)
          body <- gsub("&amp;", "&", body)
          body <- gsub("&lt;", "<", body)
          body <- gsub("&gt;", ">", body)
        }
      } else if (!is.null(part$parts)) {
        # Recursively check nested parts
        body <- paste(body, extract_body(part), sep = "\n")
      }
    }
  } else if (!is.null(payload$body$data)) {
    # Direct body data
    decoded <- base64decode(gsub("-", "+", gsub("_", "/", payload$body$data)))
    body <- rawToChar(decoded)
    if (payload$mimeType == "text/html") {
      body <- gsub("<[^>]+>", "", body)
      body <- gsub("&nbsp;", " ", body)
      body <- gsub("&amp;", "&", body)
      body <- gsub("&lt;", "<", body)
      body <- gsub("&gt;", ">", body)
    }
  }
  
  return(trimws(body))
}

# Main processing code
# Extract the messages from the gmail_messages object
if (inherits(msgs, "gmail_messages")) {
  # Access the messages list
  messages_list <- msgs[[1]]$messages
  cat(sprintf("Found %d messages to process\n", length(messages_list)))
  cat(sprintf("Note: Total messages in mailbox = %d\n", msgs[[1]]$resultSizeEstimate))
} else {
  stop("Expected a gmail_messages object")
}

# Extract all email content
email_list <- list()

# Process emails in batches to avoid rate limits
batch_size <- 10
n_messages <- length(messages_list)

for (i in seq(1, n_messages, by = batch_size)) {
  end_idx <- min(i + batch_size - 1, n_messages)
  cat(sprintf("Processing messages %d to %d of %d\n", i, end_idx, n_messages))
  
  for (j in i:end_idx) {
    email_list[[j]] <- extract_email_content(messages_list[[j]])
    
    # Small delay to avoid rate limits
    Sys.sleep(0.1)
  }
}

# Convert to data frame
emails_df <- bind_rows(email_list)

# Clean and format the data
emails_df <- emails_df %>%
  mutate(
    # Parse date to proper datetime
    date_parsed = as.POSIXct(date, format = "%a, %d %b %Y %H:%M:%S %z"),
    # Extract just email from "Name <email>" format
    from_email = str_extract(from, "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"),
    # Truncate body for display (full body still available)
    body_preview = substr(body, 1, 200)
  ) %>%
  arrange(desc(date_parsed))

# Display summary
cat("\nEmail extraction complete!\n")
cat(sprintf("Total emails processed: %d\n", nrow(emails_df)))
cat(sprintf("Date range: %s to %s\n", 
            min(emails_df$date_parsed, na.rm = TRUE),
            max(emails_df$date_parsed, na.rm = TRUE)))

# View the first few emails
print(head(emails_df %>% select(subject, from_email, date_parsed, body_preview)))

# Save to CSV for further analysis
write.csv(emails_df, "gmail_orders_extracted.csv", row.names = FALSE)

# Optional: Create a summary by thread
thread_summary <- emails_df %>%
  group_by(thread_id) %>%
  summarise(
    n_messages = n(),
    first_date = min(date_parsed, na.rm = TRUE),
    last_date = max(date_parsed, na.rm = TRUE),
    subject = first(subject),
    participants = paste(unique(from_email), collapse = ", ")
  ) %>%
  arrange(desc(last_date))

print("\nThread Summary:")
print(head(thread_summary))

# Optional: Extract specific order information if emails follow a pattern
# This is an example - adjust based on your email format
if (any(grepl("order", emails_df$subject, ignore.case = TRUE))) {
  orders_df <- emails_df %>%
    filter(grepl("order.*filled", subject, ignore.case = TRUE)) %>%
    mutate(
      # Extract order numbers if they follow a pattern (adjust regex as needed)
      order_number = str_extract(body, "Order #[0-9]+|Order Number: [0-9]+"),
      # Extract amounts if mentioned (adjust regex as needed)
      amount = str_extract(body, "\\$[0-9,]+\\.?[0-9]*")
    )
  
  print("\nOrder Information Extracted:")
  print(head(orders_df %>% select(date_parsed, from_email, order_number, amount)))
}

# Note about pagination
if (msgs[[1]]$resultSizeEstimate > n_messages) {
  cat(sprintf("\nNote: Only processed first %d messages. Total available: %d\n", 
              n_messages, msgs[[1]]$resultSizeEstimate))
  cat("To get more messages, use: gm_messages(search = 'subject:\"your order has been filled\"', page_token = msgs[[1]]$nextPageToken)\n")
}