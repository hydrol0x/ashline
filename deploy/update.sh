#!/usr/bin/env bash
# Ships your latest Ashline code from this Mac to the server and restarts it.
#   bash deploy/update.sh            (run from the repo folder, after `git pull` or your own edits)
# Your laptop's cases/ data is never copied over, so the server's live data is safe.
set -euo pipefail
SERVER="${ASHLINE_SERVER:-ubuntu@157.151.174.211}"
KEY="${ASHLINE_KEY:-$HOME/.ssh/oracle-ashline.key}"
cd "$(dirname "$0")/.."
echo "== Copying code to $SERVER"
rsync -az --delete --exclude .venv --exclude __pycache__ --exclude .cache --exclude cases --exclude .git \
  -e "ssh -i $KEY" ./ "$SERVER:~/ashline/"
echo "== Installing and restarting"
ssh -i "$KEY" "$SERVER" 'cd ~/ashline/worker && .venv/bin/pip install -q -e ".[agent,claude,heic]" && sudo systemctl restart ashline-web ashline-telegram && sleep 15 && curl -fsS -o /dev/null http://127.0.0.1:8000/ && echo "Done: the site is answering." || echo "Restarted, but the site is not answering yet: journalctl -u ashline-web -n 50"'
