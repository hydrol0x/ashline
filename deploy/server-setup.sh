#!/usr/bin/env bash
# Sets up Ashline on a fresh Ubuntu server (Oracle Cloud Always Free, Google e2-micro, any VPS).
# Runs the website and the Telegram bot as services that start on boot and restart if they crash.
#
#   1. Copy the repo, your .env and your cases/ folder to the server (see deploy/README.md).
#   2. On the server:  bash ~/ashline/deploy/server-setup.sh
#   3. Then:           sudo cloudflared service install <YOUR TUNNEL TOKEN>
#
# Safe to run again after a `git pull`: it reinstalls the package and restarts the services.
set -euo pipefail

APP="${ASHLINE_HOME:-$HOME/ashline}"
USER_NAME="$(id -un)"
[ -d "$APP/worker" ] || { echo "Expected the repo at $APP (with a worker/ folder). Set ASHLINE_HOME if it is elsewhere."; exit 1; }
[ -f "$APP/.env" ] || echo "Note: $APP/.env is missing. Copy it over before the services can work."

echo "== System packages"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git ffmpeg curl >/dev/null

echo "== Swap (small servers have 1 GB of memory; this keeps installs from running out)"
if [ "$(free -m | awk '/Mem:/{print $2}')" -lt 2000 ] && ! swapon --show | grep -q .; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

echo "== Python environment"
cd "$APP/worker"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e '.[agent,claude,heic]'

echo "== cloudflared"
if ! command -v cloudflared >/dev/null; then
  sudo mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
  echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" \
    | sudo tee /etc/apt/sources.list.d/cloudflared.list >/dev/null
  sudo apt-get update -qq && sudo apt-get install -y -qq cloudflared >/dev/null
fi

echo "== Services"
unit() {  # name, description, command
  sudo tee "/etc/systemd/system/$1.service" >/dev/null <<UNIT
[Unit]
Description=$2
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP/worker
ExecStart=$APP/worker/.venv/bin/ashline $3
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
UNIT
}
unit ashline-web "Ashline website" "serve --port 8000"
unit ashline-telegram "Ashline Telegram bot" "telegram"
sudo systemctl daemon-reload
sudo systemctl enable --now ashline-web ashline-telegram >/dev/null
sudo systemctl restart ashline-web ashline-telegram

sleep 3
echo
systemctl --no-pager --lines=0 status ashline-web ashline-telegram | grep -E "●|Active:" || true
curl -fsS -o /dev/null http://127.0.0.1:8000/ && echo "Website answers on this server." || echo "Website not answering yet: journalctl -u ashline-web -n 50"
command -v cloudflared >/dev/null && ! systemctl is-active --quiet cloudflared && \
  echo "Last step: sudo cloudflared service install <YOUR TUNNEL TOKEN>   (and stop the tunnel on your laptop)"
