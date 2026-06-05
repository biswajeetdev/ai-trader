#!/bin/bash
# Wrapper for launchd — correct Python, working dir, log rotation
cd /Users/biswajeetkumar/ai-trader

# Inject GitHub token so launchd (stripped env) can reach GitHub Models
export GITHUB_TOKEN=$(/opt/homebrew/bin/gh auth token 2>/dev/null)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

# Rotate cron.log if over 2MB
LOG=/Users/biswajeetkumar/ai-trader/cron.log
if [ -f "$LOG" ] && [ $(stat -f%z "$LOG" 2>/dev/null || echo 0) -gt 2097152 ]; then
    mv "$LOG" "${LOG}.1"
    gzip -f "${LOG}.1"
fi

/opt/homebrew/bin/python3 /Users/biswajeetkumar/ai-trader/trader.py >> "$LOG" 2>&1
