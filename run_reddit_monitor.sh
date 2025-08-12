#!/bin/bash

# Reddit Sentiment Monitor - Automated scheduler
# Run this to collect data and analyze trends

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Function to run scraper
run_scraper() {
    echo "========================================="
    echo "Running Reddit Scraper at $(date)"
    echo "========================================="
    
    Rscript reddit_sentiment_strict.R
    
    # Save timestamped copy
    if [ -f "data/traction/reddit_mentions_strict.csv" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        mkdir -p data/traction/history
        cp data/traction/reddit_mentions_strict.csv "data/traction/history/mentions_${TIMESTAMP}.csv"
        
        # Append to master history file
        if [ -f "data/traction/history/master_history.csv" ]; then
            tail -n +2 data/traction/reddit_mentions_strict.csv | \
            awk -v ts="$TIMESTAMP" -F',' '{print $0","ts}' >> data/traction/history/master_history.csv
        else
            echo "ticker,mentions,posts,comments,subreddits,timestamp" > data/traction/history/master_history.csv
            tail -n +2 data/traction/reddit_mentions_strict.csv | \
            awk -v ts="$TIMESTAMP" -F',' '{print $0","ts}' >> data/traction/history/master_history.csv
        fi
        
        echo "✅ Data saved to history"
    fi
}

# Function to analyze trends
analyze_trends() {
    echo ""
    echo "========================================="
    echo "Analyzing Trends"
    echo "========================================="
    
    # Check if we have enough history
    HISTORY_COUNT=$(find data/traction/history -name "mentions_*.csv" 2>/dev/null | wc -l)
    
    if [ "$HISTORY_COUNT" -lt 3 ]; then
        echo "⚠️  Need at least 3 data points for trend analysis"
        echo "   Current data points: $HISTORY_COUNT"
        echo "   Run the scraper more times to build history"
    else
        Rscript reddit_trend_analyzer.R analyze
    fi
}

# Function to show current top stocks
show_current() {
    echo ""
    echo "========================================="
    echo "Current Top Mentioned Stocks"
    echo "========================================="
    
    if [ -f "data/traction/reddit_mentions_strict.csv" ]; then
        echo ""
        head -11 data/traction/reddit_mentions_strict.csv | column -t -s','
        echo ""
        echo "Last updated: $(stat -c %y data/traction/reddit_mentions_strict.csv | cut -d' ' -f1,2)"
    else
        echo "No data yet. Run the scraper first."
    fi
}

# Main menu
case "${1:-menu}" in
    scrape)
        run_scraper
        ;;
    analyze)
        analyze_trends
        ;;
    both)
        run_scraper
        sleep 2
        analyze_trends
        ;;
    auto)
        # Run continuously every N hours
        INTERVAL=${2:-4}  # Default 4 hours
        echo "Starting automated monitoring (every $INTERVAL hours)"
        echo "Press Ctrl+C to stop"
        
        while true; do
            run_scraper
            show_current
            analyze_trends
            
            echo ""
            echo "Next run in $INTERVAL hours at $(date -d "+$INTERVAL hours")"
            sleep $((INTERVAL * 3600))
        done
        ;;
    setup-cron)
        # Add to crontab
        CRON_CMD="$SCRIPT_DIR/run_reddit_monitor.sh scrape"
        (crontab -l 2>/dev/null | grep -v "$CRON_CMD"; echo "0 */4 * * * $CRON_CMD") | crontab -
        echo "✅ Added to crontab - will run every 4 hours"
        crontab -l | grep reddit
        ;;
    show)
        show_current
        ;;
    *)
        echo "Reddit Sentiment Monitor"
        echo "========================"
        echo ""
        echo "Usage:"
        echo "  ./run_reddit_monitor.sh scrape    - Collect current data"
        echo "  ./run_reddit_monitor.sh analyze   - Analyze trends"  
        echo "  ./run_reddit_monitor.sh both      - Scrape then analyze"
        echo "  ./run_reddit_monitor.sh show      - Show current top stocks"
        echo "  ./run_reddit_monitor.sh auto [N]  - Run every N hours (default 4)"
        echo "  ./run_reddit_monitor.sh setup-cron - Add to crontab (every 4 hours)"
        echo ""
        show_current
        ;;
esac