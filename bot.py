#!/usr/bin/env python3
"""
Market News -> Telegram bot (AI-filtered).

Checks major market news feeds (stocks, macro, forex, commodities + a little crypto),
asks an AI to keep ONLY the genuinely market-moving items, and posts a short formatted
alert to Telegram. Everything else is dropped.

Runs on a schedule (e.g. GitHub Actions). State is kept in seen.json.

Required environment variables / GitHub secrets:
  TELEGRAM_BOT_TOKEN  - from @BotFather
  TELEGRAM_CHAT_ID    - your numeric chat or channel id
  AI_API_KEY          - an API key for an OpenAI-compatible provider (Groq has a free tier)

Optional (have sensible defaults for Groq's free tier):
  AI_BASE_URL         - default https://api.groq.com/openai/v1
  AI_MODEL            - default llama-3.3-70b-versatile
"""

import os
import sys
import json
import time
import html
from pathlib import Path

import feedparser
import requests

# ---------------------------------------------------------------------------
# 1. News sources (all markets). The AI keeps only the important ones, so it's
#    fine to include broad feeds here. Add/remove lines freely.
# ---------------------------------------------------------------------------
FEEDS = {
    # Stocks / general markets
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
    # Macro / economy / central banks
    "Bloomberg Economics": "https://feeds.bloomberg.com/economics/news.rss",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
    # Forex
    "ForexLive": "https://www.forexlive.com/feed/",
    # Commodities / energy
    "OilPrice": "https://oilprice.com/rss/main",
    # Crypto (background)
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
}

# ---------------------------------------------------------------------------
# 2. Settings you can tweak
# ---------------------------------------------------------------------------
MAX_AI_CALLS = 10         # max articles judged by the AI per run (protects free-tier limits)
SEEN_KEEP = 4000          # how many article ids to remember
SLEEP_BETWEEN_AI = 2.5    # seconds between AI calls (stay under free rate limits)
SLEEP_BETWEEN_SENDS = 1.1 # seconds between Telegram messages
SUMMARY_CHARS = 400       # how much of each article we feed the AI

SEEN_FILE = Path(__file__).with_name("seen.json")
PROMPT_FILE = Path(__file__).with_name("filter_prompt.txt")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
TG_API = f"https://api.telegram.org/bot{TOKEN}"

AI_KEY = os.environ.get("AI_API_KEY", "").strip()
AI_BASE_URL = os.environ.get("AI_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")
AI_MODEL = os.environ.get("AI_MODEL", "llama-3.3-70b-versatile").strip()

SYSTEM_PROMPT = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""


# ---------------------------------------------------------------------------
# Seen-state helpers
# ---------------------------------------------------------------------------
def load_seen():
    if not SEEN_FILE.exists():
        return set(), True
    try:
        data = json.loads(SEEN_FILE.read_text() or "[]")
        ids = set(data)
        return ids, len(ids) == 0
    except Exception:
        return set(), True


def save_seen(ids):
    SEEN_FILE.write_text(json.dumps(list(ids)[-SEEN_KEEP:], ensure_ascii=False))


def article_id(entry):
    return entry.get("id") or entry.get("guid") or entry.get("link") or entry.get("title", "")


def clean(text):
    # strip HTML tags / whitespace from summaries
    import re
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = html.unescape(text)
    return " ".join(text.split())


def fetch_all():
    """Return list of dicts for every current item across all feeds."""
    items = []
    for source, url in FEEDS.items():
        try:
            parsed = feedparser.parse(url)
        except Exception as e:
            print(f"[warn] could not fetch {source}: {e}", file=sys.stderr)
            continue
        if parsed.bozo and not parsed.entries:
            print(f"[warn] {source} returned no usable entries", file=sys.stderr)
            continue
        for entry in parsed.entries:
            uid = article_id(entry)
            if not uid:
                continue
            t = entry.get("published_parsed") or entry.get("updated_parsed")
            sort_time = time.mktime(t) if t else time.time()
            items.append({
                "uid": uid,
                "time": sort_time,
                "source": source,
                "title": clean(entry.get("title", "(no title)")),
                "summary": clean(entry.get("summary", ""))[:SUMMARY_CHARS],
                "link": entry.get("link", ""),
            })
    return items


# ---------------------------------------------------------------------------
# AI filter (OpenAI-compatible chat completions; works with Groq / OpenAI / etc.)
# ---------------------------------------------------------------------------
def ai_classify(item):
    """Return alert text to send, or None to skip. Raises on a hard API error."""
    user = (
        f"Title: {item['title']}\n"
        f"Summary: {item['summary']}\n"
        f"Source: {item['source']}"
    )
    resp = requests.post(
        f"{AI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {AI_KEY}", "Content-Type": "application/json"},
        json={
            "model": AI_MODEL,
            "temperature": 0,
            "max_tokens": 220,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )
    if resp.status_code == 429:
        raise RuntimeError("ai_rate_limited")
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()
    if not text or text.upper().startswith("SKIP") or text.upper() == "SKIP":
        return None
    return text


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
def telegram_send(text):
    resp = requests.post(
        f"{TG_API}/sendMessage",
        json={"chat_id": CHAT_ID, "text": text, "disable_web_page_preview": False},
        timeout=30,
    )
    if resp.status_code == 429:
        retry = resp.json().get("parameters", {}).get("retry_after", 5)
        time.sleep(retry + 1)
        return telegram_send(text)
    if not resp.ok:
        print(f"[error] telegram {resp.status_code}: {resp.text}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    missing = [k for k, v in {
        "TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": CHAT_ID, "AI_API_KEY": AI_KEY,
    }.items() if not v]
    if missing:
        print(f"[fatal] missing secrets: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    seen, first_run = load_seen()
    items = fetch_all()
    if not items:
        print("[info] no items fetched")
        return

    # First run: seed everything (no AI), just confirm it's live.
    if first_run:
        for it in items:
            seen.add(it["uid"])
        save_seen(seen)
        telegram_send(
            "✅ Market news bot is live. Watching: " + ", ".join(FEEDS.keys())
            + ".\nYou'll get only the market-moving items from now on."
        )
        print(f"[info] first run: seeded {len(items)} existing articles")
        return

    # New articles, newest first (judge the freshest if we hit the cap).
    new_items = [it for it in items if it["uid"] not in seen]
    new_items.sort(key=lambda x: x["time"], reverse=True)
    if not new_items:
        print("[info] no new articles")
        return

    judged = new_items[:MAX_AI_CALLS]
    # anything beyond the cap is marked seen silently (avoid backlog buildup)
    for it in new_items[MAX_AI_CALLS:]:
        seen.add(it["uid"])

    to_send = []  # (time, alert_text, link) for items the AI kept
    for it in judged:
        try:
            alert = ai_classify(it)
        except Exception as e:
            # rate-limited or API error: stop here, leave remaining unseen for next run
            print(f"[warn] AI stopped: {e}", file=sys.stderr)
            break
        seen.add(it["uid"])  # judged -> don't re-judge next run
        if alert:
            to_send.append((it["time"], alert, it["source"], it["link"]))
        time.sleep(SLEEP_BETWEEN_AI)

    # Send kept alerts oldest-first so they arrive in order.
    to_send.sort(key=lambda x: x[0])
    sent = 0
    for _, alert, source, link in to_send:
        msg = f"{alert}\n🗞 {source}"
        if link:
            msg += f"\n{link}"
        if telegram_send(msg):
            sent += 1
            time.sleep(SLEEP_BETWEEN_SENDS)

    save_seen(seen)
    print(f"[info] judged {len(judged)}, sent {sent} alert(s)")


if __name__ == "__main__":
    main()
