# Deployment Guide
## NQ/ES Trade Decision Dashboard — Complete Setup from Zero

---

## What you'll have at the end

| Platform | URL | What it does |
|---|---|---|
| **GitHub Pages** | `https://YOUR_NAME.github.io/nq-es-dashboard` | Static dashboard, always online, free |
| **Streamlit Cloud** | `https://YOUR_APP.streamlit.app` | Full Python engine, CSV upload, live data ready |
| **Railway** (later) | `https://xyz.railway.app` | Webhook receiver for TradingView live data |

---

# PART 1 — GitHub Pages (Static Dashboard)

This puts `index.html` online permanently for free.
No coding. No server. Works in 10 minutes.

---

## Step 1 — Rename the dashboard file

On your computer, find `nq_es_decision.html` and rename it to:

```
index.html
```

That is the only file GitHub Pages needs to find your site automatically.

---

## Step 2 — Create a GitHub repository

1. Open your browser and go to **https://github.com**
2. Sign in to your account
3. Click the **green "New"** button (top left, next to your profile)
4. Fill in:
   - **Repository name:** `nq-es-dashboard`  ← type exactly this
   - **Description:** `NQ/ES futures trade decision dashboard`
   - **Public** ← must be Public for free GitHub Pages
   - **Do NOT** tick "Add a README file" (we already have one)
5. Click **"Create repository"** (green button at the bottom)

You'll see an empty repository page. Leave it open.

---

## Step 3 — Upload your files

1. On your new empty repository page, click **"uploading an existing file"**
   (it's a link in the middle of the page)
2. Drag and drop ALL these files onto the upload area:
   ```
   index.html                          ← the renamed dashboard
   README.md
   nq_es_strategy.py
   streamlit_app.py
   requirements.txt
   ```
3. Scroll down, type a commit message: `Initial upload`
4. Click **"Commit changes"** (green button)

You'll see all your files listed in the repository. ✓

---

## Step 4 — Enable GitHub Pages

1. Click **"Settings"** (top menu bar of your repository)
2. In the left sidebar, scroll down and click **"Pages"**
3. Under **"Source"**, click the dropdown that says "None"
4. Select **"Deploy from a branch"**
5. Under **"Branch"**, select **"main"**
6. Leave the folder as `/ (root)`
7. Click **"Save"**

GitHub will show: *"Your site is being deployed…"*

---

## Step 5 — Wait 60–90 seconds, then visit your site

Refresh the Settings → Pages page after a minute.
You'll see:

```
✓ Your site is live at https://YOUR_USERNAME.github.io/nq-es-dashboard
```

Click the link. Your dashboard is now live on the internet. ✓

> **Every time you upload new files to GitHub**, the site updates automatically
> within about 60 seconds. No further steps needed.

---

## Step 6 — Enable auto-deploy (optional but recommended)

This uses the `.github/workflows/deploy.yml` file to auto-rebuild on every push.

1. In your repository, click **"Actions"** (top menu)
2. You'll see a workflow called **"Deploy to GitHub Pages"**
3. If it shows a yellow dot (running) — it's working. Wait for the green tick ✓
4. From now on, any file you change and commit will be automatically deployed

---

---

# PART 2 — Streamlit Cloud (Python Dashboard)

This runs the full Python strategy engine online for free.
Required for: CSV upload, live API data, auto-refresh.

---

## Step 1 — Make sure these files are in your GitHub repo

```
streamlit_app.py        ← main app
nq_es_strategy.py       ← strategy engine (imported by the app)
requirements.txt        ← Python packages
api/
  __init__.py           ← empty file (needed so Python treats api/ as a package)
  topstep_connector.py
  tradingview_connector.py
```

Create the empty `__init__.py` file:
1. In your GitHub repository, click **"Add file"** → **"Create new file"**
2. Name it: `api/__init__.py`
3. Leave it blank
4. Click **"Commit new file"**

---

## Step 2 — Sign up / log in to Streamlit Cloud

1. Go to **https://share.streamlit.app**
2. Click **"Sign up"**
3. Sign up **with GitHub** (this links your repositories automatically)

---

## Step 3 — Deploy your app

1. Once logged in, click **"New app"** (top right)
2. Fill in:
   - **Repository:** `YOUR_USERNAME/nq-es-dashboard`  ← select from dropdown
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Click **"Deploy!"**

Streamlit will install your requirements and start the app.
It takes 2–3 minutes the first time.

4. Your app URL will be something like:
   ```
   https://nq-es-dashboard-abc123.streamlit.app
   ```

---

## Step 4 — Test it

1. Open your Streamlit URL
2. You'll see the dashboard with "Upload CSV files" mode selected
3. Upload `NQ_30min_sample.csv` and `ES_30min_sample.csv` from your local data
4. The strategy runs and you see the decision card

If you see an error, click "Manage app" → "Logs" to read the error message.

---

## Step 5 — Update your README with the live URLs

1. Open `README.md` in your GitHub repository
2. Replace the placeholder URLs:
   ```
   https://YOUR_USERNAME.github.io/nq-es-dashboard
   https://YOUR_APP.streamlit.app
   ```
   with your actual URLs
3. Commit the change

---

---

# PART 3 — Connecting Topstep (Live Data)

Topstep uses **Tradovate** as its trading platform.
Tradovate has an API that provides real-time bar data.

---

## Step 1 — Enable API access in Tradovate

1. Log into your Topstep account
2. Open the Tradovate platform (the trading platform Topstep uses)
3. Go to **Settings** → **API Access**
4. Enable API access
5. Note your:
   - Username (your Topstep email)
   - Password (your Topstep password)

---

## Step 2 — Add credentials to Streamlit Secrets

Never put your password in a code file. Use Streamlit Secrets instead.

1. Go to **https://share.streamlit.app**
2. Find your app → click the **three dots (⋮)** → **"Settings"**
3. Click **"Secrets"**
4. Paste this (fill in your real values):

```toml
TOPSTEP_USERNAME = "your.email@example.com"
TOPSTEP_PASSWORD = "yourpassword"
TOPSTEP_DEMO = true
```

5. Click **"Save"**

> `TOPSTEP_DEMO = true` means it uses Topstep's simulation account.
> Change to `false` only when you are ready to use your funded account.

---

## Step 3 — Test the connection locally first

On your own computer, open a terminal and run:

```bash
# Set credentials temporarily (Mac/Linux)
export TOPSTEP_USERNAME="your.email@example.com"
export TOPSTEP_PASSWORD="yourpassword"

# Test the connector
python api/topstep_connector.py
```

On Windows (Command Prompt):
```cmd
set TOPSTEP_USERNAME=your.email@example.com
set TOPSTEP_PASSWORD=yourpassword
python api/topstep_connector.py
```

You should see 10 bars of NQ and ES data printed to the terminal.
If it works locally, it will work in Streamlit too.

---

## Step 4 — Update requirements.txt

Open `requirements.txt` and uncomment the `requests` line (it's probably already there).
Also uncomment `websocket-client` if you want live tick streaming later:

```
requests>=2.31.0
websocket-client>=1.7.0
```

Push the change to GitHub. Streamlit will reinstall packages automatically.

---

## Step 5 — Use live data in the dashboard

1. Open your Streamlit app
2. In the left sidebar, change **Data Source** to **"🔴 Topstep (live)"**
3. Select your timeframe (30min recommended to start)
4. Enable **Auto-refresh** — set to 30 seconds for 30-min bars

The dashboard will now fetch live data from Tradovate every 30 seconds
and update the decision card automatically.

---

## What to do when the front-month contract rolls

NQ and ES contracts roll quarterly:
- March contract (H): expires 3rd Friday of March
- June contract (M): expires 3rd Friday of June
- September contract (U): expires 3rd Friday of September
- December contract (Z): expires 3rd Friday of December

When the contract rolls, update `SYMBOL_MAP` in `api/topstep_connector.py`:

```python
SYMBOL_MAP = {
    "NQ": "NQZ4",   # December 2024
    "ES": "ESZ4",
}
```

Then push to GitHub. Streamlit redeploys automatically.

---

---

# PART 4 — Connecting TradingView (Webhook Data)

TradingView sends bar data to a URL every time a new bar closes.
This is the simplest live data option if you already use TradingView.

---

## Step 1 — Deploy the webhook receiver to Railway

Railway is a free hosting service for small Python apps.

1. Go to **https://railway.app**
2. Sign up with GitHub
3. Click **"New Project"** → **"Deploy from GitHub repo"**
4. Select your `nq-es-dashboard` repository
5. Railway will detect it's a Python app

6. Set the **Start Command** to:
   ```
   python api/tradingview_connector.py --serve
   ```

7. Click **"Deploy"**

Railway gives you a public URL like:
```
https://nq-es-dashboard-production.up.railway.app
```

Your webhook URL is:
```
https://nq-es-dashboard-production.up.railway.app/webhook
```

Write this URL down — you need it in TradingView.

---

## Step 2 — Add the Pine Script to TradingView

1. Open **TradingView** → open an **NQ1!** chart
2. At the bottom, click **"Pine Script Editor"**
3. Click **"Open"** → **"New indicator"**
4. Delete all the existing code
5. Copy the Pine Script from `api/tradingview_connector.py`
   (it's in the `PINE_SCRIPT_TEMPLATE` string at the top of the file)
6. Paste it into the editor
7. Click **"Save"** → name it `NQ/ES Webhook Feed`
8. Click **"Add to chart"**

Now do the **same for ES1!** — open an ES1! chart and add the same script.

---

## Step 3 — Create the TradingView alert

1. On your NQ1! chart, click the **"Alert"** button (clock icon, top right)
2. Fill in:
   - **Condition:** select your `NQ/ES Webhook Feed` script
   - **Alert name:** `NQ Bar Feed`
   - Scroll down to **"Webhook URL"**
   - Paste your Railway URL:
     ```
     https://YOUR_RAILWAY_URL/webhook
     ```
   - **Message:** leave as-is (the Pine Script fills it in)
3. Set **Expiration** to a date far in the future (e.g. 1 year)
4. Click **"Create"**

Repeat for ES1! — create a second alert with the same webhook URL.

---

## Step 4 — Verify bars are arriving

1. Open your browser and visit:
   ```
   https://YOUR_RAILWAY_URL/bars/NQ/30min
   ```
2. After the next 30-minute bar closes on TradingView, you'll see bar data:
   ```json
   {"symbol":"NQ","tf":"30min","count":1,"bars":[...]}
   ```

---

## Step 5 — Connect the webhook store to Streamlit

The `tradingview_connector.py` saves bars to `api/tv_bars.json`.
Streamlit reads this file via `get_latest_tv_bars()`.

For Streamlit Cloud to read data from Railway, you have two options:

**Option A (simple) — Shared file storage via GitHub:**
The webhook server commits `tv_bars.json` to your repository every N bars.
Streamlit reads it from disk. Set up via Railway's GitHub integration.

**Option B (recommended) — Direct API call:**
Streamlit fetches bars from the Railway `/bars/` endpoint directly:

In `streamlit_app.py`, replace `load_from_tradingview()` with:
```python
import requests

def load_from_tradingview(tf):
    railway_url = st.secrets["RAILWAY_URL"]
    nq = requests.get(f"{railway_url}/bars/NQ/{tf}").json()["bars"]
    es = requests.get(f"{railway_url}/bars/ES/{tf}").json()["bars"]
    return pd.DataFrame(nq), pd.DataFrame(es)
```

Then add to Streamlit Secrets:
```toml
RAILWAY_URL = "https://YOUR_RAILWAY_URL.railway.app"
```

---

---

# PART 5 — Auto-Alerts (Signal Notifications)

When a signal fires, you want to know immediately — not just when you check the dashboard.

## Email alerts via Gmail

Add this to `streamlit_app.py` after `render_decision()`:

```python
import smtplib
from email.message import EmailMessage

def send_email_alert(signal: str, entry: float, sl: float, tp: float):
    msg = EmailMessage()
    msg["Subject"] = f"🚨 NQ Signal: {signal}"
    msg["From"]    = st.secrets["ALERT_EMAIL"]
    msg["To"]      = st.secrets["ALERT_EMAIL"]
    msg.set_content(
        f"Signal: {signal}\n"
        f"Entry:  {entry:,.2f}\n"
        f"SL:     {sl:,.2f}\n"
        f"TP:     {tp:,.2f}\n"
    )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(st.secrets["ALERT_EMAIL"], st.secrets["ALERT_PASSWORD"])
        smtp.send_message(msg)
```

Add to Streamlit Secrets:
```toml
ALERT_EMAIL    = "your@gmail.com"
ALERT_PASSWORD = "your_gmail_app_password"   # Use Gmail App Password, not your real password
```

To get a Gmail App Password:
1. Go to https://myaccount.google.com/security
2. Enable 2-Step Verification
3. Search for "App passwords" → create one → copy the 16-character code

---

## Telegram alerts (even better — instant on your phone)

```python
import requests

def send_telegram(signal: str, entry: float, sl: float, tp: float):
    token   = st.secrets["TELEGRAM_BOT_TOKEN"]
    chat_id = st.secrets["TELEGRAM_CHAT_ID"]
    text = (
        f"📈 *NQ/ES Signal*\n"
        f"*{signal}*\n\n"
        f"Entry: `{entry:,.2f}`\n"
        f"SL:    `{sl:,.2f}`\n"
        f"TP:    `{tp:,.2f}`\n"
    )
    requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    )
```

To set up a Telegram bot (5 minutes):
1. Open Telegram → search for **@BotFather**
2. Type `/newbot` → follow the prompts → copy the token
3. Start a chat with your new bot
4. Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
5. Copy the `chat_id` number from the response

Add to Streamlit Secrets:
```toml
TELEGRAM_BOT_TOKEN = "123456:ABCdef..."
TELEGRAM_CHAT_ID   = "987654321"
```

---

---

# Quick Reference — File Summary

| File | Purpose | Where it runs |
|---|---|---|
| `index.html` | Static dashboard, sample data | GitHub Pages (browser) |
| `streamlit_app.py` | Full app with CSV upload + live data | Streamlit Cloud |
| `nq_es_strategy.py` | Core strategy engine + CLI tool | Local / any server |
| `api/topstep_connector.py` | Tradovate/Topstep data feed | Streamlit Cloud |
| `api/tradingview_connector.py` | TradingView webhook receiver | Railway.app |
| `requirements.txt` | Python package list | Streamlit Cloud, Railway |
| `.github/workflows/deploy.yml` | Auto-deploy GitHub Pages | GitHub Actions |
| `README.md` | Project documentation | GitHub |

---

# Troubleshooting

**GitHub Pages shows a 404 error**
→ Make sure the file is named exactly `index.html` (not `Index.html` or `index.HTML`)
→ Wait 2 minutes after enabling Pages and refresh

**Streamlit app shows a module not found error**
→ Check `requirements.txt` has all needed packages
→ Click "Manage app" → "Reboot app" in Streamlit Cloud

**Topstep connector says "auth error"**
→ Double-check your credentials in Streamlit Secrets
→ Make sure API access is enabled in your Tradovate account settings
→ Try `TOPSTEP_DEMO = true` first before going live

**TradingView alerts not arriving**
→ Check the webhook URL has no typos
→ Visit `https://YOUR_RAILWAY_URL/` — it should show `{"status":"running"}`
→ TradingView only sends webhook alerts when a bar *closes*, not on every tick
→ Check your TradingView alert is set to "Once Per Bar Close"

**No signals showing**
→ Signals only fire when all 5 conditions are true simultaneously
→ Try 5-min or 1-min timeframes — more bars = more signal opportunities
→ Try uploading more data (more date range = better warm-up for EMAs and correlation)

---

*Questions or issues: open a GitHub Issue on your repository*
