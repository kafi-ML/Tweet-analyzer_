#!/usr/bin/env python3
"""
preprocess_tweets.py
Improved token extraction pipeline:
 - Layer 1: high-recall regex candidate extraction (cashtags, hashtags, contracts, pairs)
 - Layer 2: context heuristics (crypto keyword window, stopwords)
 - Layer 3 (optional): semantic verification with a zero-shot classifier (enable with --use_nlp)

Usage example:
  python preprocess_tweets.py --input data/parsed_tweets.jsonl --out_cleaned data/cleaned_tweets.jsonl \
    --out_tokens data/token_mentions.jsonl --limit 100 --overwrite --verbose

Notes:
 - By default Layer 3 (transformers) is OFF. Enable with --use_nlp (requires transformers + torch).
"""

import argparse
import json
import os
import logging
import math
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

# Optional modules
try:
    from langdetect import detect as lang_detect
except Exception:
    lang_detect = None

try:
    import spacy
    nlp_spacy = spacy.load("en_core_web_sm")
except Exception:
    nlp_spacy = None

try:
    # Zero-shot classifier from transformers (optional)
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    pipeline = None
    _HAS_TRANSFORMERS = False

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE if (BASE / "data").exists() else Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "preprocess.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("preprocess")

# Regex patterns (expanded)
# require at least one letter in cashtag (avoid $2, $5, $36B etc)
CASH_TAG_RE = re.compile(r"\$([A-Za-z][A-Za-z0-9]{0,11})")
HASHTAG_RE = re.compile(r"#([A-Za-z0-9_]{1,64})")
HEX_CONTRACT_RE = re.compile(r"(0x[a-fA-F0-9]{40})")
PAIR_RE = re.compile(r"\b([A-Z0-9]{2,8})[\/\-]([A-Z0-9]{2,8})\b")
ALLCAPS_TOKEN_RE = re.compile(r"\b[A-Z]{2,8}\b")

# Crypto-context keywords (for heuristic)
CRYPTO_KEYWORDS = {
    "token","coin","buy","sell","hold","hodl","airdrop","presale","launch","listing","listed",
    "dex","liquidity","lp","chart","pump","moon","rug","mint","pair","swap","contract",
    "staking","uniswap","pancake","dexscreener","etherscan","solscan","contract","tokenomics"
}

# Common noise stopwords to ignore as tokens
STOPWORDS_UPPER = {
    "RT","THE","AND","FOR","YOU","WITH","WHEN","INTO","THIS","THAT","FROM","HERE","WHAT",
    "IN","ON","TO","BY","AS","AT","IS","ARE","WILL","NEW","NOW"
}
STOPWORDS_UPPER.update({"CAN","SEE","ME","SHOW","NOT","BE","AM","PT","ALL","CEO","RIP","THEY","WE","I","YOU"})

# Manager-tunable defaults
DEFAULT_DEDUPE_SECONDS = 60

def iso_to_dt(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

def to_iso_utc(dt: datetime):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def normalize_token(tok: str):
    if tok is None:
        return tok
    tok = str(tok).strip()
    # contract remained lowercase
    if tok.lower().startswith("0x"):
        return tok.lower()
    # strip punctuation and leading $ or #
    tok = tok.lstrip("$#").upper()
    return tok

def weighted_engagement(likes, retweets, quotes, replies, followers):
    base = (likes or 0) + 2 * (retweets or 0) + 3 * (quotes or 0) + (replies or 0)
    return base * math.log1p(max(0, followers or 0))

def hash_key(text: str):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# ---------------- Token candidate extraction ----------------
def extract_candidates_from_text(text, raw_obj=None):
    """
    High-recall candidate extraction from tweet text & raw entities:
      - cashtags ($TOK)
      - hashtags (#TOK)
      - contract addresses (0x...)
      - token pairs (PEPE/ETH)
      - uppercase words fallback (but filtered later)
    Returns: candidates_set, metadata_list
    metadata_list contains dicts with source hints for debugging
    """
    candidates = []
    meta = []

    if not text:
        return [], []

    # cashtags
    for m in CASH_TAG_RE.findall(text):
        t = normalize_token(m)
        candidates.append(t)
        meta.append({"candidate": t, "source": "cashtag"})

    # hashtags
    for m in HASHTAG_RE.findall(text):
        t = normalize_token(m)
        candidates.append(t)
        meta.append({"candidate": t, "source": "hashtag"})

    # contract addresses in text
    for m in HEX_CONTRACT_RE.findall(text):
        t = m.lower()
        candidates.append(t)
        meta.append({"candidate": t, "source": "contract"})

    # pairs like DOGE/ETH or DOGE-ETH
    for a,b in PAIR_RE.findall(text):
        t = normalize_token(a)
        candidates.append(t)
        meta.append({"candidate": t, "source": "pair"})
        t2 = normalize_token(b)
        candidates.append(t2)
        meta.append({"candidate": t2, "source": "pair"})

    # check raw entities if present
    if raw_obj:
        # legacy entities urls
        urls = safe_get(raw_obj, "legacy", "entities", "urls") or safe_get(raw_obj, "entities", "urls") or []
        for u in urls:
            if isinstance(u, dict):
                eu = u.get("expanded_url") or u.get("url")
                if eu:
                    # contract in url
                    for m in HEX_CONTRACT_RE.findall(eu):
                        candidates.append(m.lower()); meta.append({"candidate": m.lower(), "source": "contract_url"})
                    # try to extract symbol-like path segments
                    parts = [p for p in re.split(r"[\/\?\=&#]+", eu) if p]
                    # heuristic: short alpha path segments may be token symbol
                    for p in parts[::-1][:6]:
                        if re.match(r"^[A-Za-z]{2,8}$", p):
                            candidates.append(normalize_token(p)); meta.append({"candidate": normalize_token(p), "source": "url_path"})
    # uppercase fallback
     # uppercase fallback — collect but mark source 'allcaps' (we'll validate later)
    for m in ALLCAPS_TOKEN_RE.findall(text):
        if m in STOPWORDS_UPPER:
            continue
        # skip short 1-letter tokens
        if len(m) <= 1:
            continue
        # skip pure numeric-ish tokens (e.g., '16M', '45M') if they are mostly digits with suffix
        if re.match(r"^\d+[MK]?$", m):
            continue
        candidates.append(normalize_token(m))
        meta.append({"candidate": normalize_token(m), "source": "allcaps"})


    # deduplicate maintain order
    seen = set()
    uniq = []
    uniq_meta = []
    for i in meta:
        c = i["candidate"]
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
        uniq_meta.append(i)
    return uniq, uniq_meta

# ---------------- Context heuristics ----------------
def has_crypto_context(text, window_tokens=10):
    """
    Quick heuristic: check if the 10-word window around the candidate contains crypto keywords
    or punctuation patterns (., $, 0x).
    """
    if not text:
        return False
    lower = text.lower()
    # quick keyword presence
    for k in CRYPTO_KEYWORDS:
        if k in lower:
            return True
    # look for $ anywhere
    if "$" in text:
        return True
    # look for 0x contract mention
    if "0x" in text.lower():
        return True
    return False

# ---------------- Semantic verification (optional) ----------------
def build_nlp_verifier(model_name="facebook/bart-large-mnli"):
    """
    Build a zero-shot classification pipeline.
    This is optional and heavy; call only if --use_nlp True.
    """
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("Transformers not installed. pip install transformers torch")
    # create pipeline
    classifier = pipeline("zero-shot-classification", model=model_name)
    return classifier

def semantic_is_crypto(classifier, text):
    """
    Use zero-shot classification to determine if the tweet is crypto-related.
    Labels: 'cryptocurrency', 'non-crypto'
    Returns probability for 'cryptocurrency' label (0..1)
    """
    # candidate labels
    try:
        out = classifier(text, candidate_labels=["cryptocurrency","non-crypto"], hypothesis_template="This text is about {}.")
        # out['labels'] sorted highest-first
        # find 'cryptocurrency'
        for label, score in zip(out["labels"], out["scores"]):
            if label == "cryptocurrency":
                return float(score)
        # fallback
        return float(out["scores"][0])
    except Exception:
        return 0.0

# ---------------- Main processing ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Preprocess parsed_tweets -> cleaned + token mentions")
    p.add_argument("--input", default=str(DATA_DIR / "parsed_tweets.jsonl"))
    p.add_argument("--out_cleaned", default=str(DATA_DIR / "cleaned_tweets.jsonl"))
    p.add_argument("--out_tokens", default=str(DATA_DIR / "token_mentions.jsonl"))
    p.add_argument("--min_followers", type=int, default=0)
    p.add_argument("--lang", default="en")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dedupe_window", type=int, default=DEFAULT_DEDUPE_SECONDS)
    p.add_argument("--use_nlp", action="store_true", help="Enable semantic verification (requires transformers + torch)")
    p.add_argument("--nlp_model", default="facebook/bart-large-mnli", help="Transformers model for zero-shot (only if --use_nlp)")
    return p.parse_args()

def main():
    args = parse_args()
    infile = Path(args.input)
    out_clean = Path(args.out_cleaned)
    out_tok = Path(args.out_tokens)

    if not infile.exists():
        print("Input file not found:", infile)
        sys.exit(1)

    if args.overwrite:
        if out_clean.exists(): out_clean.unlink()
        if out_tok.exists(): out_tok.unlink()

    # optional NLP classifier
    classifier = None
    if args.use_nlp:
        if not _HAS_TRANSFORMERS:
            print("Error: transformers not installed. Install with 'pip install transformers torch' and try again.")
            sys.exit(1)
        print("Loading NLP verifier model (this may take a while)...")
        classifier = build_nlp_verifier(args.nlp_model)
        print("NLP verifier ready.")

    processed = 0
    saved_clean = 0
    saved_tokens = 0
    spam_count = 0
    dup_count = 0

    dedupe_map = {}  # dedupe_key -> last_seen datetime

    # repetition detector
    user_text_counter = defaultdict(Counter)

    with infile.open("r", encoding="utf-8") as fin, \
         out_clean.open("a", encoding="utf-8") as fclean, \
         out_tok.open("a", encoding="utf-8") as ftok:
        for line in fin:
            if args.limit and processed >= args.limit:
                break
            processed += 1
            try:
                rec = json.loads(line)
            except Exception:
                logger.exception("Skipping invalid JSON line")
                continue

            tid = rec.get("id") or safe_get(rec, "_raw", "rest_id") or rec.get("tweet_id")
            text = rec.get("text") or safe_get(rec, "_raw", "legacy", "full_text") or ""
            created_raw = rec.get("created_at") or safe_get(rec, "_raw", "legacy", "created_at")
            dt = iso_to_dt(created_raw) or datetime.now(timezone.utc)
            created_iso = to_iso_utc(dt)

            username = rec.get("username") or safe_get(rec, "_raw", "core", "user_results", "result", "legacy", "screen_name") or ""
            display_name = rec.get("display_name") or safe_get(rec, "_raw", "core", "user_results", "result", "legacy", "name") or ""
            user_id = rec.get("user_id") or safe_get(rec, "_raw", "core", "user_results", "result", "rest_id") or ""

            followers = int(rec.get("followers_count") or safe_get(rec, "_raw", "core", "user_results", "result", "legacy", "followers_count") or 0)
            likes = int(rec.get("likes") or rec.get("favorite_count") or safe_get(rec, "_raw", "legacy", "favorite_count") or 0)
            retweets = int(rec.get("retweets") or rec.get("retweet_count") or safe_get(rec, "_raw", "legacy", "retweet_count") or 0)
            replies = int(rec.get("replies") or rec.get("reply_count") or safe_get(rec, "_raw", "legacy", "reply_count") or 0)
            quote_count = int(rec.get("quote_count") or safe_get(rec, "_raw", "legacy", "quote_count") or 0)
            views = int(rec.get("views") or safe_get(rec, "_raw", "views", "count") or 0)

            # get raw entity urls list if present
            raw = rec.get("_raw") or {}
            entities_urls = []
            try:
                ent_urls = safe_get(raw, "legacy", "entities", "urls") or safe_get(raw, "entities", "urls") or []
                if isinstance(ent_urls, list):
                    for u in ent_urls:
                        if isinstance(u, dict):
                            u_e = u.get("expanded_url") or u.get("url")
                            if u_e: entities_urls.append(u_e)
            except Exception:
                pass
            # fallback rec['urls']
            if not entities_urls and rec.get("urls"):
                for u in rec.get("urls"):
                    if isinstance(u, str):
                        entities_urls.append(u)

            # media parse
            media = rec.get("media") or []
            # compute weighted engagement
            we = weighted_engagement(likes, retweets, quote_count, replies, followers)

            # dedupe by text+username
            dedupe_key = hash_key((text or "") + "::" + username)
            last_dt = dedupe_map.get(dedupe_key)
            window = timedelta(seconds=args.dedupe_window)
            if last_dt and (dt - last_dt) <= window:
                dup_count += 1
                continue
            dedupe_map[dedupe_key] = dt
            user_text_counter[username][text] += 1

            # spam heuristics (light) - but we will not drop verified authors by default
            is_spam = False
            if len(text.strip()) < 4 and entities_urls:
                is_spam = True
            if len(entities_urls) > 4:
                is_spam = True
            if user_text_counter[username][text] > 6:
                is_spam = True
            if followers < args.min_followers:
                is_spam = True

            # language check (best-effort)
            lang = None
            try:
                lang = safe_get(rec, "_raw", "legacy", "lang") or rec.get("lang")
                if not lang and lang_detect and text.strip():
                    lang = lang_detect(text)
            except Exception:
                lang = None
            if args.lang != "all" and lang and lang[:2].lower() != args.lang[:2].lower():
                # mark but do not drop necessarily
                is_spam = True

            # build cleaned record
            cleaned = {
                "id": tid,
                "text": text,
                "created_at": created_iso,
                "username": username,
                "display_name": display_name,
                "user_id": user_id,
                "followers_count": followers,
                "likes": likes,
                "retweets": retweets,
                "replies": replies,
                "quote_count": quote_count,
                "views": views,
                "urls": entities_urls,
                "media": media,
                "weighted_engagement": we,
                "is_reply": bool(rec.get("is_reply")),
                "in_reply_to": rec.get("in_reply_to"),
                "spam": bool(is_spam),
                "collected_at": rec.get("collected_at") or datetime.now(timezone.utc).isoformat()
            }

            # write cleaned
            fclean.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            saved_clean += 1

            # ---------- token extraction + validation ----------
            candidates, meta = extract_candidates_from_text(text, raw)
            if args.verbose:
                print(f"[DEBUG] candidates: {candidates} meta: {meta}")

            # Quick semantic check if enabled: if classifier says not crypto, we will *not* accept any tokens
            semantic_score = None
            if classifier:
                # keep text size reasonable
                score = semantic_is_crypto(classifier, text[:1000])
                semantic_score = score
                if args.verbose:
                    print(f"[DEBUG] semantic crypto score: {score:.3f}")
                # if score is very low, skip token writing (but keep cleaned tweet)
                if score < 0.25:
                    # count as "no token due semantic"
                    if args.verbose:
                        print(f"[SKIP TOKENS] semantic score too low ({score:.3f}) for tweet id {tid}")
                    continue

            # validate each candidate using heuristics — but because you monitor verified accounts,
            # accept more aggressively. Preference order: cashtag, contract, url-derived, pair, uppercase + context
            accepted = []
            for c in candidates:
                if not c:
                    continue

                # 1) contracts always accepted
                if isinstance(c, str) and c.lower().startswith("0x"):
                    accepted.append(c.lower()); continue

                # 2) strong-signal meta acceptance
                if any(m["source"] in ("cashtag","hashtag","contract","contract_url","url_path","pair") and m["candidate"]==c for m in meta):
                    accepted.append(c); continue

                # 3) if the tweet contains dexscreener/etherscan/solscan -> accept (URL hint)
                if entities_urls and any(("dexscreener" in u.lower() or "etherscan" in u.lower() or "solscan" in u.lower() or "bscscan" in u.lower()) for u in entities_urls):
                    accepted.append(c); continue

                # 4) If candidate came from 'allcaps', require crypto context OR spaCy PROPN/NER signal
                src = next((m["source"] for m in meta if m["candidate"]==c), None)
                if src == "allcaps":
                    ctx_ok = has_crypto_context(text)
                    spacy_ok = False
                    try:
                        if nlp_spacy:
                            doc = nlp_spacy(text)
                            # check if candidate appears as PROPN or named entity (ORG/PRODUCT) in doc
                            for token in doc:
                                if token.text.upper() == c and token.pos_ == "PROPN":
                                    spacy_ok = True
                                    break
                            for ent in doc.ents:
                                if ent.text.upper() == c and ent.label_ in ("ORG","PRODUCT","NORP"):
                                    spacy_ok = True
                                    break
                    except Exception:
                        spacy_ok = False

                    if ctx_ok or spacy_ok:
                        accepted.append(c)
                        continue
                    # else: skip this allcaps candidate (no context + not PROPN)
                    if args.verbose:
                        print(f"[SKIP] allcaps candidate '{c}' skipped (no crypto context and not PROPN/NER).")
                    continue

                # 5) fallback: if has crypto context anywhere in tweet accept
                if has_crypto_context(text):
                    accepted.append(c); continue

                # 6) last resort: (since verified authors) accept but log (this keeps recall high)
                accepted.append(c)
                if args.verbose:
                    print(f"[WARN] fallback accept candidate '{c}' (no strong signal).")


            # dedupe accepted tokens
            accepted = list(dict.fromkeys(accepted))
            # remove obvious meaningless tokens like 'RT' or 'THE'
            accepted = [a for a in accepted if a and a.upper() not in STOPWORDS_UPPER]

            # write token mentions
            for tok in accepted:
                token_norm = normalize_token(tok)
                token_record = {
                    "token": token_norm,
                    "tweet_id": tid,
                    "username": username,
                    "created_at": created_iso,
                    "weighted_engagement": we,
                    "followers_count": followers,
                    "is_reply": bool(rec.get("is_reply")),
                    "is_retweet": bool(safe_get(rec, "_raw", "legacy", "retweeted") or False),
                    "source": "twikit",
                    "semantic_score": semantic_score
                }
                ftok.write(json.dumps(token_record, ensure_ascii=False) + "\n")
                saved_tokens += 1

            if is_spam:
                spam_count += 1

            if args.verbose and processed % 200 == 0:
                print(f"Processed {processed}, saved_clean {saved_clean}, saved_tokens {saved_tokens}, spam {spam_count}, dup {dup_count}")

    summary = {
        "processed": processed,
        "saved_clean": saved_clean,
        "saved_tokens": saved_tokens,
        "spam_count": spam_count,
        "duplicates_dropped": dup_count,
        "cleaned_path": str(out_clean.resolve()),
        "tokens_path": str(out_tok.resolve())
    }
    print("Done:", summary)
    logger.info("Preprocess summary: %s", summary)

if __name__ == "__main__":
    main()
