# scraper/twikit_collector.py
# Twikit-only data collector for MemeRadar
# Usage:
#   python -m scraper.twikit_collector --user elonmusk --count 5
#   python -m scraper.twikit_collector --accounts configs/accounts.txt --count 5
#
# Requirements: twikit (v2.x), install in your venv:
#   pip install twikit

import argparse
import asyncio
import json
import os
import logging
import sys
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from twikit import Client

# CONFIG
COOKIE_FILE = "x_cookies.json"           # must contain {"auth_token": "...", "ct0": "..."}
ACCOUNTS_FILE = "configs/accounts.txt"   # optional: one username per line (no @)
RAW_OUTPUT = "data/raw_tweets.jsonl"
PARSED_OUTPUT = "data/parsed_tweets.jsonl"
ACCOUNTS_OUTPUT = "data/accounts.jsonl"
TWEETS_PER_USER = 5                      # default now set to 5 (latest posts)

# Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "twikit_collector.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("twikit_collector")


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("configs", exist_ok=True)


def load_accounts_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    # strip leading @ if present
    return [ln.lstrip("@") for ln in lines]


def save_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_get(obj: Any, *attrs, default=None):
    """Safe attribute access: try attributes chain, fallback to dict keys if possible."""
    try:
        cur = obj
        for a in attrs:
            if cur is None:
                return default
            # If hasattr and attribute is present, use it
            if hasattr(cur, a):
                cur = getattr(cur, a)
            elif isinstance(cur, dict) and a in cur:
                cur = cur[a]
            else:
                return default
        return cur
    except Exception:
        return default


# ---------- robust parsing helpers ----------
def get_from_nested(d: dict, path_list, default=None):
    """Safely get nested keys from a dict-like object following a list path split by '.' or list input."""
    if d is None:
        return default
    cur = d
    if isinstance(path_list, str):
        path_list = path_list.split(".")
    try:
        for p in path_list:
            if cur is None:
                return default
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur
    except Exception:
        return default


def parse_datetime_str(dt_str: str):
    """Parse Twitter-style created_at like 'Fri Sep 05 16:37:20 +0000 2025' to ISO8601 UTC."""
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc).isoformat()
    except Exception:
        # fallback: try ISO parse or return original string
        try:
            return datetime.fromisoformat(dt_str).astimezone(timezone.utc).isoformat()
        except Exception:
            return dt_str


def normalize_int(v):
    try:
        if v is None:
            return 0
        if isinstance(v, int):
            return v
        s = str(v)
        m = re.search(r"\d+", s.replace(",", ""))
        return int(m.group(0)) if m else 0
    except Exception:
        return 0


def extract_urls_from_raw(raw: dict, tweet_obj=None):
    """Return list of expanded URLs from common raw locations."""
    out = []
    try:
        # 1) tweet entities (legacy)
        urls = get_from_nested(raw, "legacy.entities.urls") or get_from_nested(raw, "entities.urls")
        if isinstance(urls, list):
            for u in urls:
                if isinstance(u, dict):
                    out.append(u.get("expanded_url") or u.get("url"))
        # 2) check other fields (entities top-level)
        if not out and tweet_obj is not None:
            ent = getattr(tweet_obj, "entities", None) or (getattr(tweet_obj, "_data", {}) or {}).get("entities", {})
            if isinstance(ent, dict):
                for u in ent.get("urls", []):
                    if isinstance(u, dict):
                        out.append(u.get("expanded_url") or u.get("url"))
        # dedupe & clean
        out = [u for u in (out or []) if u]
        # final cleaning: remove t.co-only duplicates if media exists
        return list(dict.fromkeys(out))
    except Exception:
        return []


def extract_media_from_raw(raw: dict, tweet_obj=None):
    """Return list of media dicts (type, url). Look in extended_entities or legacy entities."""
    media_out = []
    try:
        # 1) check legacy extended_entities
        mlist = get_from_nested(raw, "legacy.extended_entities.media") or get_from_nested(raw, "extended_entities.media")
        if not mlist:
            mlist = get_from_nested(raw, "legacy.entities.media") or get_from_nested(raw, "entities.media")
        if isinstance(mlist, list):
            for m in mlist:
                if not isinstance(m, dict):
                    continue
                url = m.get("media_url_https") or m.get("media_url") or m.get("url") or m.get("preview_image_url")
                mtype = m.get("type") or m.get("media_type") or "photo"
                if url:
                    media_out.append({"type": mtype, "url": url})
        # 2) fallback: check tweet_obj.media attribute if present
        if not media_out and tweet_obj is not None and hasattr(tweet_obj, "media"):
            try:
                for m in getattr(tweet_obj, "media") or []:
                    if isinstance(m, dict):
                        media_out.append({"type": m.get("type", "media"), "url": m.get("media_url_https") or m.get("url")})
            except Exception:
                pass
        # dedupe
        seen = set()
        result = []
        for m in media_out:
            if m["url"] not in seen:
                seen.add(m["url"])
                result.append(m)
        return result
    except Exception:
        return []


def find_display_name(raw: dict, user_obj=None):
    """Return a robust display name: prefer user_obj, else nested raw legacy name."""
    name = None
    if user_obj is not None:
        name = getattr(user_obj, "name", None) or getattr(user_obj, "display_name", None) or getattr(user_obj, "full_name", None)
    if not name and raw:
        name = get_from_nested(raw, "core.user_results.result.legacy.name") or get_from_nested(raw, "legacy.user.name") or get_from_nested(raw, "user.name")
    return name or ""


def find_username(raw: dict, user_obj=None):
    uname = None
    if user_obj is not None:
        uname = getattr(user_obj, "screen_name", None) or getattr(user_obj, "username", None)
    if not uname and raw:
        uname = get_from_nested(raw, "core.user_results.result.legacy.screen_name") or get_from_nested(raw, "legacy.user.screen_name") or get_from_nested(raw, "user.screen_name")
    return (uname or "").lstrip("@")


def find_followers(raw: dict, user_obj=None):
    if user_obj is not None:
        v = getattr(user_obj, "followers_count", None) or getattr(user_obj, "followers", None)
        if v is not None:
            return normalize_int(v)
    # Try nested legacy path
    v = get_from_nested(raw, "core.user_results.result.legacy.followers_count")
    if v is None:
        v = get_from_nested(raw, "legacy.user.followers_count") or get_from_nested(raw, "legacy.followers_count") or get_from_nested(raw, "user.followers_count")
    return normalize_int(v)


def find_views(raw: dict):
    v = get_from_nested(raw, "views.count") or get_from_nested(raw, "views")
    return normalize_int(v)


# ---------- build_parsed_record ----------
def build_parsed_record(tweet: Any, user_obj: Any) -> Dict[str, Any]:
    """Assemble a robust normalized parsed record using tweet attributes and raw dict fallback."""
    # try to obtain raw dict
    raw_json = None
    if hasattr(tweet, "raw"):
        raw_json = getattr(tweet, "raw")
        if isinstance(raw_json, str):
            try:
                raw_json = json.loads(raw_json)
            except Exception:
                raw_json = {"raw": raw_json}
    elif hasattr(tweet, "_data"):
        raw_json = getattr(tweet, "_data")
    else:
        raw_json = None

    # primary simple fields
    tid = getattr(tweet, "id", None) or get_from_nested(raw_json or {}, "rest_id") or get_from_nested(raw_json or {}, "legacy.id_str")
    # text: prefer tweet.text or legacy.full_text / legacy.full_text / legacy.full_text (different shapes)
    text = getattr(tweet, "text", None) or get_from_nested(raw_json or {}, "legacy.full_text") or get_from_nested(raw_json or {}, "legacy.full_text") or get_from_nested(raw_json or {}, "legacy.display_text_range") or ""
    # created_at: prefer tweet.created_at else legacy.created_at
    created_raw = getattr(tweet, "created_at", None) or get_from_nested(raw_json or {}, "legacy.created_at") or get_from_nested(raw_json or {}, "created_at")
    created_at = parse_datetime_str(created_raw) if isinstance(created_raw, str) else (str(created_raw) if created_raw else None)

    username = find_username(raw_json or {}, user_obj)
    display_name = find_display_name(raw_json or {}, user_obj)
    user_id = getattr(user_obj, "id", None) or get_from_nested(raw_json or {}, "core.user_results.result.rest_id") or get_from_nested(raw_json or {}, "core.user_results.result.legacy.id_str") or get_from_nested(raw_json or {}, "legacy.user.id_str")

    followers = find_followers(raw_json or {}, user_obj)

    # engagement counts: prefer exposed fields; fall back to legacy
    likes = normalize_int(getattr(tweet, "favorite_count", None) or get_from_nested(raw_json or {}, "legacy.favorite_count") or get_from_nested(raw_json or {}, "favorite_count"))
    retweets = normalize_int(getattr(tweet, "retweet_count", None) or get_from_nested(raw_json or {}, "legacy.retweet_count") or get_from_nested(raw_json or {}, "retweet_count"))
    replies = normalize_int(getattr(tweet, "reply_count", None) or get_from_nested(raw_json or {}, "legacy.reply_count") or get_from_nested(raw_json or {}, "reply_count"))
    quote_count = normalize_int(get_from_nested(raw_json or {}, "legacy.quote_count") or get_from_nested(raw_json or {}, "quote_count"))

    # referenced tweets (quoted / replied)
    referenced = get_from_nested(raw_json or {}, "legacy.referenced_tweets") or get_from_nested(raw_json or {}, "referenced_tweets") or []

    urls = extract_urls_from_raw(raw_json or {}, tweet)
    media = extract_media_from_raw(raw_json or {}, tweet)

    in_reply_to = get_from_nested(raw_json or {}, "legacy.in_reply_to_status_id_str") or get_from_nested(raw_json or {}, "in_reply_to_status_id") or None
    is_reply = bool(in_reply_to)

    views = find_views(raw_json or {})

    parsed = {
        "id": tid,
        "text": text or "",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "username": username or "",
        "display_name": display_name or "",
        "user_id": user_id,
        "followers_count": followers,
        "likes": likes,
        "retweets": retweets,
        "replies": replies,
        "quote_count": quote_count,
        "views": views,
        "urls": urls,
        "media": media,
        "referenced_tweets": referenced,
        "is_reply": is_reply,
        "in_reply_to": in_reply_to,
        "source": "twikit",
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }

    if raw_json:
        parsed["_raw"] = raw_json

    return parsed


def ensure_cookie_file():
    if not os.path.exists(COOKIE_FILE):
        logger.error("Cookie file missing: %s", COOKIE_FILE)
        print("❌ Cookie file missing: ", COOKIE_FILE)
        sys.exit(1)


def load_client() -> Client:
    # client is synchronous in creation but Twikit methods are async
    c = Client(language="en-US")
    # try load cookies via Twikit, fallback to setting via dict if needed
    try:
        c.load_cookies(COOKIE_FILE)
        logger.info("Loaded cookies via client.load_cookies()")
    except Exception as e:
        logger.warning("client.load_cookies failed: %s", e)
        # fallback: if JSON contains auth_token and ct0 try to set them
        try:
            with open(COOKIE_FILE, "r", encoding="utf-8") as f:
                cookie_data = json.load(f)
            if isinstance(cookie_data, dict):
                for k, v in cookie_data.items():
                    try:
                        c.set_cookie(k, v)
                    except Exception:
                        # ignore if set_cookie not available
                        pass
                logger.info("Manually set cookie keys into client.")
            else:
                logger.error("Cookie file format unexpected.")
                print("❌ Cookie file format unexpected. It should be a JSON dict with auth_token and ct0.")
                sys.exit(1)
        except Exception as e2:
            logger.exception("Failed to read cookie file fallback: %s", e2)
            print("❌ Failed to read cookie file fallback:", e2)
            sys.exit(1)
    return c


async def collect_for_user(client: Client, username: str, count: int):
    """Collect tweets for a single username and write to raw & parsed outputs."""
    username = username.lstrip("@")
    logger.info("Collecting for user: %s", username)
    try:
        user = await client.get_user_by_screen_name(username)
    except Exception as e:
        logger.error("Failed to get user %s: %s", username, e)
        print(f"❌ Failed to get user {username}: {e}")
        return

    # Save user metadata to accounts file (line per user)
    try:
        acct = {
            "user_id": getattr(user, "id", None) or safe_get(user, "id"),
            "username": getattr(user, "screen_name", None) or getattr(user, "username", None) or username,
            "display_name": getattr(user, "name", None) or getattr(user, "display_name", None),
            "followers_count": normalize_int(getattr(user, "followers_count", None) or safe_get(user, "followers_count", 0)),
            "created_at": str(getattr(user, "created_at", None) or safe_get(user, "created_at", None)),
            "collected_at": datetime.now(timezone.utc).isoformat()
        }
        save_jsonl(ACCOUNTS_OUTPUT, acct)
    except Exception as e:
        logger.warning("Failed to save account metadata for %s: %s", username, e)

    # Fetch tweets (async)
    try:
        tweets = await user.get_tweets("Tweets", count=count)
    except Exception as e:
        logger.error("Failed to fetch tweets for %s: %s", username, e)
        print(f"❌ Failed to fetch tweets for {username}: {e}")
        return

    # Write raw and parsed data
    raw_written = 0
    parsed_written = 0
    for t in tweets:
        # Try to get a raw dict if available
        raw_obj = None
        if hasattr(t, "raw"):
            raw_obj = getattr(t, "raw")
            # if string try to parse
            if isinstance(raw_obj, str):
                try:
                    raw_obj = json.loads(raw_obj)
                except Exception:
                    raw_obj = {"raw": raw_obj}
        elif hasattr(t, "_data"):
            raw_obj = getattr(t, "_data", None)

        # Save raw (best-effort)
        raw_record = {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "username": username,
            "source": "twikit",
            "tweet_obj": raw_obj if raw_obj is not None else {
                "id": getattr(t, "id", None), "text": getattr(t, "text", None)
            }
        }
        save_jsonl(RAW_OUTPUT, raw_record)
        raw_written += 1

        # Build parsed and save
        parsed = build_parsed_record(t, user)
        save_jsonl(PARSED_OUTPUT, parsed)
        parsed_written += 1

    logger.info("Finished user %s: raw=%d parsed=%d", username, raw_written, parsed_written)
    print(f"Saved {parsed_written} parsed tweets and {raw_written} raw tweets for @{username}")


async def run_collect(usernames: List[str], per_user: int):
    ensure_dirs()
    ensure_cookie_file()
    client = load_client()
    # iterate sequentially to avoid triggering blocks; you can parallelize later
    for u in usernames:
        await collect_for_user(client, u, per_user)


def parse_args():
    p = argparse.ArgumentParser(description="Twikit-only tweet collector")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--user", help="single username to collect (without @)")
    group.add_argument("--accounts", help=f"path to accounts file (one username per line). default: {ACCOUNTS_FILE}")
    p.add_argument("--count", type=int, default=TWEETS_PER_USER, help="tweets per user to fetch")
    return p.parse_args()


def main():
    args = parse_args()
    if args.user:
        users = [args.user.lstrip("@")]
    else:
        path = args.accounts or ACCOUNTS_FILE
        users = load_accounts_from_file(path)
        if not users:
            print("No accounts found in", path)
            sys.exit(1)
    try:
        asyncio.run(run_collect(users, args.count))
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", e)
        print("Unhandled error:", e)


if __name__ == "__main__":
    main()
