# Script to fix environment variable naming mismatch

# Load the existing .env file
if (file.exists("config/.env")) {
  readRenviron("config/.env")
  
  # Map old names to new names
  # The app expects: STORAGE_ACCOUNT_NAME, ACCESS_KEY, CONTAINER_NAME
  # But .env has: AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_KEY, AZURE_CONTAINER_NAME
  
  # Set the expected variable names from the existing ones
  Sys.setenv(
    STORAGE_ACCOUNT_NAME = Sys.getenv("AZURE_STORAGE_ACCOUNT"),
    ACCESS_KEY = Sys.getenv("AZURE_STORAGE_KEY"),
    CONTAINER_NAME = Sys.getenv("AZURE_CONTAINER_NAME")
  )
  
  cat("Environment variables mapped:\n")
  cat("AZURE_STORAGE_ACCOUNT -> STORAGE_ACCOUNT_NAME\n")
  cat("AZURE_STORAGE_KEY -> ACCESS_KEY\n")
  cat("AZURE_CONTAINER_NAME -> CONTAINER_NAME\n")
} else {
  cat("config/.env not found!\n")
}