# Setup script to install all R dependencies
packages <- c('reticulate', 'arrow', 'AzureStor', 'dplyr', 'readr', 'lubridate')

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = 'https://cloud.r-project.org/')
  }
}

cat("All R packages installed successfully\n")