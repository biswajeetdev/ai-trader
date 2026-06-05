#!/bin/bash
# server_setup.sh — Run once on a fresh Ubuntu 22.04 VM (Oracle Free Tier)
# Usage: bash server_setup.sh
set -e

echo "=== AI-Trader 24/7 Server Setup ==="

# 1. System packages
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv git gh logrotate

# 2. Clone / pull repo
REPO_DIR="$HOME/ai-trader"
if [ ! -d "$REPO_DIR" ]; then
    echo "Copy your ai-trader folder to this server, then re-run."
    echo "  scp -r ~/ai-trader ubuntu@YOUR_SERVER_IP:~/"
    exit 1
fi

# 3. Python venv + deps
cd "$REPO_DIR"
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 4. GitHub CLI auth (for GitHub Models LLM free tier)
echo "Log in to GitHub CLI (needed for LLM brain):"
gh auth login

# 5. Logs directory
mkdir -p logs

# 6. Install systemd service + timer
sudo cp scripts/trader.service /etc/systemd/system/ai-trader.service
sudo cp scripts/trader.timer   /etc/systemd/system/ai-trader.timer

# Fix user in service file
sudo sed -i "s/User=ubuntu/User=$USER/" /etc/systemd/system/ai-trader.service
sudo sed -i "s|/home/ubuntu|$HOME|g"   /etc/systemd/system/ai-trader.service

sudo systemctl daemon-reload
sudo systemctl enable ai-trader.timer
sudo systemctl start  ai-trader.timer

# 7. Log rotation (keep 7 days)
sudo tee /etc/logrotate.d/ai-trader > /dev/null <<EOF
$HOME/ai-trader/logs/trader.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
EOF

echo ""
echo "=== Done! ==="
echo ""
echo "Bot runs every 30 min, 24/7."
echo "Crypto trades any time. Stocks auto-gate to market hours."
echo ""
echo "Useful commands:"
echo "  systemctl status  ai-trader.timer    # is it scheduled?"
echo "  systemctl list-timers ai-trader*     # next run time"
echo "  journalctl -u ai-trader.service -f   # live logs"
echo "  tail -f ~/ai-trader/logs/trader.log  # full output log"
echo "  systemctl stop ai-trader.timer       # pause the bot"
echo "  systemctl start ai-trader.timer      # resume"
