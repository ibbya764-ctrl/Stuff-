# Going Public — Deployment Guide

## The fastest path: Cloudflare Tunnel

Cloudflare Tunnel exposes your local server to the internet securely.
- Free on Cloudflare's free plan
- SSL/HTTPS handled automatically
- DDoS protection included
- No router port-forwarding needed
- Works from behind any firewall

---

## Step 1 — Install the Scaffold dependencies for public mode

Open a terminal in your scaffold folder and run:

    pip install pyjwt bcrypt python-multipart

---

## Step 2 — Install Cloudflare Tunnel (cloudflared)

### On Windows:
Download the installer from:
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.msi

Run the MSI installer. After installation, open a new terminal and verify:
  cloudflared --version

### On Linux/Mac:
  # Linux
  wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x cloudflared-linux-amd64
  sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared

---

## Step 3 — Authenticate with Cloudflare

    cloudflared tunnel login

This opens a browser. Log in to your Cloudflare account (free account works).
If you don't have one: https://dash.cloudflare.com/sign-up

---

## Step 4 — Create the tunnel

    cloudflared tunnel create scaffold-ai

This gives you a tunnel ID. Note it down.

---

## Step 5 — Create the tunnel config file

Create a file called `cloudflared.yml` in your scaffold folder:

    tunnel: YOUR-TUNNEL-ID-HERE
    credentials-file: C:\Users\YOURUSERNAME\.cloudflared\YOUR-TUNNEL-ID-HERE.json

    ingress:
      - hostname: YOUR-DOMAIN-OR-SUBDOMAIN
        service: http://localhost:8000
      - service: http_status:404

Replace:
  YOUR-TUNNEL-ID-HERE  → the tunnel ID from step 4
  YOURUSERNAME         → your Windows username
  YOUR-DOMAIN-OR-SUBDOMAIN → either a subdomain on a domain you own,
                               or a free Cloudflare subdomain

For a free subdomain without a domain:
  cloudflared tunnel route dns scaffold-ai scaffold-ai.pages.dev
  (This may vary — Cloudflare's free options change. Check their docs.)

If you have your own domain (e.g., yoursite.com):
  hostname: scaffold.yoursite.com
  Then run: cloudflared tunnel route dns scaffold-ai scaffold.yoursite.com

---

## Step 6 — Start everything

Open THREE terminal windows:

### Terminal 1 — Ollama
  ollama serve

### Terminal 2 — The Scaffold
  cd C:\Users\YOURUSERNAME\Desktop\scaffold
  set SCAFFOLD_SECRET=your-random-secret-key-change-this
  python web_api.py

### Terminal 3 — Cloudflare Tunnel
  cd C:\Users\YOURUSERNAME\Desktop\scaffold
  cloudflared tunnel --config cloudflared.yml run

The scaffold is now publicly accessible at your tunnel URL.

---

## Step 7 — Create the first admin account

The first time web_api.py starts with no users, it prints an invite code:
  [auth] No users found. Admin invite code: XXXXXXXXXXXXXXXX
  [auth] Register at /register with this code.

Note this code. Go to your public URL and register with it.
You'll be the admin user.

To create invite codes for others, use the admin endpoint:
  POST /admin/invite  (requires admin auth token)
  {"role": "user", "limit": 1}

---

## Running automatically on Windows startup

To have everything start automatically when Windows boots:

1. Create a batch file called `start_scaffold.bat` in your scaffold folder:

    @echo off
    start "" ollama serve
    timeout /t 5
    cd /d C:\Users\YOURUSERNAME\Desktop\scaffold
    set SCAFFOLD_SECRET=your-secret-key-here
    start "" python web_api.py
    timeout /t 5
    cloudflared tunnel --config cloudflared.yml run

2. Press Win+R, type `shell:startup`, press Enter
3. Copy `start_scaffold.bat` into the startup folder that opens

Everything will start automatically when you log in.

---

## Expected capacity

With qwen2.5:7b on the Lenovo (CPU inference):
  ~2-5 concurrent users comfortable
  ~10-20 users with queuing (2-5 min wait)

To handle more users without waiting:
  Set ANTHROPIC_API_KEY environment variable
  The system automatically uses Claude API for inference
  Much faster, handles unlimited concurrent users
  Cost: ~£0.002-0.005 per question with Haiku

---

## Environment variables (set before running web_api.py)

  SCAFFOLD_SECRET      — random secret for JWT tokens (required)
  ANTHROPIC_API_KEY    — Claude API key (optional, for cloud inference)
  RATE_LIMIT_HOUR      — questions per user per hour (default: 20)
  RATE_LIMIT_DAY       — questions per user per day (default: 100)

---

## Monitoring

Visit your-url/status  — live system statistics
Visit your-url/admin   — admin panel (admin users only)

The admin panel shows:
  - Active users and their usage
  - Queue status
  - Collective learning stats
  - Belief system state
  - DMN integration level (Φ)
