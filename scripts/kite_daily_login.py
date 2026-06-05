#!/usr/bin/env python3
"""
kite_daily_login.py — Run once every morning before 9:15 AM IST.

Usage:
    python3 scripts/kite_daily_login.py

Requirements:
    pip install pyotp requests

Config keys in config.json:
    "zerodha_user_id":     "ZQ1234",
    "zerodha_password":    "yourpassword",
    "zerodha_totp_secret": "BASE32_FROM_2FA_APP"
"""

import json, sys
from pathlib import Path

DIR    = Path(__file__).parent.parent
CONFIG = DIR / "config.json"


def main():
    if not CONFIG.exists():
        print("ERROR: config.json not found")
        sys.exit(1)

    cfg     = json.loads(CONFIG.read_text())
    missing = [k for k in ["zerodha_user_id", "zerodha_password", "zerodha_totp_secret"]
               if not cfg.get(k)]
    if missing:
        print("Missing config keys:", missing)
        print("\nAdd to config.json:")
        print('  "zerodha_user_id":     "ZQ1234",')
        print('  "zerodha_password":    "your_zerodha_password",')
        print('  "zerodha_totp_secret": "BASE32_FROM_AUTHENTICATOR_APP"')
        print('\nGet TOTP secret: in your 2FA app → "Cannot scan?" → copy the 32-char code')
        sys.exit(1)

    try:
        import pyotp  # noqa
    except ImportError:
        print("Run: pip install pyotp requests")
        sys.exit(1)

    print("Logging into Zerodha via TOTP...")
    try:
        from broker.zerodha_enctoken import login
        token = login(cfg)
        print(f"  enctoken saved → zerodha_token.json")
        print(f"  Valid until 6 AM tomorrow. Bot will use it automatically.")
    except Exception as e:
        print(f"  Login failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
