# Running Ashline on a free Oracle Cloud server

This moves the website and the Telegram bot off your laptop onto Oracle Cloud's Always Free tier. ashline.work keeps working through the same Cloudflare tunnel, so the domain and DNS don't change and no ports need opening.

Takes about 30–45 minutes the first time.

## 1. Create the Oracle account

1. Sign up at https://signup.oraclecloud.com.
2. It asks for a phone number and a credit card for identity checks. The card is not charged unless you upgrade the account. Prepaid and virtual cards are refused.
3. **Choose the home region carefully: it can't be changed later**, and free servers only run there. Pick a US region. If one is often "out of capacity", that's why people pick a less busy one (for example US Midwest (Chicago) rather than US East (Ashburn)).
4. **Never click "Upgrade to Pay As You Go"** if you want to stay at $0. The free account starts with a 30-day trial; when it ends, Always Free resources keep running.

## 2. Create the server

In the Oracle console: menu → **Compute → Instances → Create instance**.

- **Name:** `ashline`
- **Image:** click *Edit* → *Change image* → **Canonical Ubuntu 24.04**.
- **Shape:** *Change shape* → **Ampere** → **VM.Standard.A1.Flex**, 2 OCPUs and 12 GB memory. It must show the *Always Free-eligible* label.
  - "Out of capacity"? Try 1 OCPU / 6 GB, try again later, or use **VM.Standard.E2.1.Micro** (AMD, 1 GB, also Always Free). The setup script adds swap so 1 GB is enough.
- **Networking:** leave the defaults; make sure *Assign a public IPv4 address* is on.
- **SSH keys:** choose *Generate a key pair for me* and **download the private key**. You can't download it again later.
- Click **Create** and wait until it says *Running*. Copy its **Public IP address**.

## 3. Connect from your Mac

```bash
mv ~/Downloads/ssh-key-*.key ~/.ssh/oracle-ashline.key
chmod 600 ~/.ssh/oracle-ashline.key
ssh -i ~/.ssh/oracle-ashline.key ubuntu@<PUBLIC_IP>
```

Type `yes` the first time. Type `exit` to come back to your Mac.

## 4. Stop Ashline on your laptop

Ctrl-C `ashline serve`, `ashline telegram` and the `cloudflared` tunnel. Two copies of the Telegram bot fight over messages, and two tunnels split visitors between machines.

## 5. Copy Ashline to the server (from your Mac)

This copies the code, your `.env` and your `cases/` data. Nothing goes through GitHub, so no tokens are needed:

```bash
rsync -az --exclude .venv --exclude __pycache__ --exclude .cache \
  -e "ssh -i ~/.ssh/oracle-ashline.key" ~/ashline/ ubuntu@<PUBLIC_IP>:~/ashline/
```

## 6. Run the setup script (on the server)

```bash
ssh -i ~/.ssh/oracle-ashline.key ubuntu@<PUBLIC_IP>
bash ~/ashline/deploy/server-setup.sh
```

It installs Python, ffmpeg and cloudflared, installs Ashline, and creates two services (`ashline-web`, `ashline-telegram`) that start on boot and restart if they crash. It ends by checking the website answers.

## 7. Connect the tunnel (on the server)

Get the tunnel token: Cloudflare dashboard → **Networking → Tunnels** → your tunnel → the install/connector instructions show a command with a long token. Then:

```bash
sudo cloudflared service install <TOKEN>
```

The tunnel already points ashline.work at `http://127.0.0.1:8000`, so the site comes back within a minute.

## 8. Check it

- Open https://ashline.work. It should show the Telegram button.
- Message the bot on Telegram. It should reply.

## Everyday commands (on the server)

```bash
journalctl -u ashline-telegram -f        # watch the bot's log (Ctrl-C to stop watching)
journalctl -u ashline-web -n 50          # the website's last 50 log lines
sudo systemctl restart ashline-web ashline-telegram
```

## Updating after code changes

From the repo folder on your Mac, after `git pull` (or your own edits):

```bash
bash deploy/update.sh
```

It copies the code (never your laptop's `cases/` data), reinstalls, restarts both services and checks the site. Takes about a minute.

## Things to know

- **Idle reclaim:** Oracle may reclaim Always Free servers that stay idle for a long time. If yours disappears, recreate it (steps 2–7). Keep a copy of `cases/` by occasionally copying it back:
  `rsync -az -e "ssh -i ~/.ssh/oracle-ashline.key" ubuntu@<PUBLIC_IP>:~/ashline/cases/ ~/ashline-backup/`
- **API costs:** every photo sent to the bot uses Anthropic credits. Set a monthly spending limit in the Anthropic console.
- **Telegram works here** even though campus wifi blocks it: the server isn't on campus wifi.
