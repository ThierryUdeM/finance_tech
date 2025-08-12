# Changelog

## [1.0.1] - 2024-08-12

### Fixed
- Updated GitHub Actions to use latest versions:
  - `actions/upload-artifact@v3` → `v4`
  - `actions/cache@v3` → `v4`
- Fixed YAML syntax errors in workflow files (heredoc issues)

### Security
- All actions now use latest supported versions
- Removed deprecated action versions

## [1.0.0] - 2024-08-12

### Added
- Reddit sentiment scanner with NASDAQ validation
- GitHub Actions workflows for automated monitoring
- Azure Storage integration for data persistence
- Trend analysis and reporting capabilities
- Documentation for setup and usage

### Features
- Scrapes r/wallstreetbets, r/stocks, r/investing
- Validates tickers against official NASDAQ/NYSE lists
- Runs automatically every 4 hours
- Creates GitHub issues for high-momentum stocks
- Generates multiple report types (summary, detailed, trending, historical)