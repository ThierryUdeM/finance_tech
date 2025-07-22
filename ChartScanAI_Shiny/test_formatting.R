# Test formatting functions

# Test prices
test_prices <- c(104567.89, 234.56, 12.34, 0.99, 1000000)

cat("Testing price formatting:\n")
for (price in test_prices) {
  # This is what we're using in the app
  formatted <- paste0("$", formatC(price, format = "f", digits = 2, big.mark = ","))
  cat(sprintf("  %.2f -> %s\n", price, formatted))
}

# Alternative formatting options if needed
cat("\nAlternative formatting with prettyNum:\n")
for (price in test_prices) {
  formatted <- paste0("$", prettyNum(round(price, 2), big.mark = ",", scientific = FALSE))
  cat(sprintf("  %.2f -> %s\n", price, formatted))
}

# Format using scales package (if available)
if (requireNamespace("scales", quietly = TRUE)) {
  cat("\nUsing scales package:\n")
  for (price in test_prices) {
    formatted <- scales::dollar(price)
    cat(sprintf("  %.2f -> %s\n", price, formatted))
  }
}