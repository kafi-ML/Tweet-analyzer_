# analyzer/compute_metrics.py  (updated, robust)
import os, json, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from scipy.stats import entropy

DATA_PATH = "data/token_mentions.jsonl"
OUTPUT_PATH = "data/token_metrics.jsonl"
TIME_WINDOW_HOURS = 1      # window size (hours)
BASELINE_WINDOWS = 6       # number of previous windows to form baseline
EPS = 1e-6
MAX_RATIO = 1e6            # safety cap for extreme ratios

def load_data(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        raise ValueError("No input rows found in " + path)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at", "token"])
    df["token"] = df["token"].astype(str).str.upper()
    # ensure numeric column
    df["weighted_engagement"] = pd.to_numeric(df.get("weighted_engagement", 0), errors="coerce").fillna(0)
    return df

def aggregate_window(df, start_time, end_time):
    dfw = df[(df["created_at"] >= start_time) & (df["created_at"] < end_time)]
    out = []
    if dfw.empty:
        return pd.DataFrame(out)
    for token, g in dfw.groupby("token"):
        mentions = len(g)
        unique_authors = g["username"].nunique() if "username" in g else g.get("user_id", pd.Series()).nunique()
        avg_engagement = float(g["weighted_engagement"].mean()) if not g["weighted_engagement"].empty else 0.0
        # author entropy (safe)
        counts = g["username"].value_counts(normalize=True) if "username" in g else pd.Series()
        author_entropy = float(entropy(counts, base=2)) if len(counts) > 0 else 0.0
        out.append({
            "token": token,
            "window_start": start_time,
            "window_end": end_time,
            "mentions": int(mentions),
            "unique_authors": int(unique_authors),
            "avg_engagement": float(avg_engagement),
            "author_entropy": float(author_entropy)
        })
    return pd.DataFrame(out)

def compute_metrics(df_all):
    df_all = df_all.sort_values("created_at")
    min_time, max_time = df_all["created_at"].min(), df_all["created_at"].max()
    window_delta = timedelta(hours=TIME_WINDOW_HOURS)

    # Build all windows
    windows = []
    cursor = min_time.floor(freq=f"{TIME_WINDOW_HOURS}H") if hasattr(min_time, "floor") else min_time
    while cursor < max_time + window_delta:
        start = cursor
        end = start + window_delta
        agg = aggregate_window(df_all, start, end)
        if not agg.empty:
            windows.append(agg)
        cursor = end

    print(f"[INFO] Total {len(windows)} windows aggregated")

    # Build time-indexed list for baseline lookup
    token_history = {}  # token -> list of dicts per window (in order)
    results = []

    for w_idx, win_df in enumerate(windows):
        # append this window's token rows to token_history
        for _, row in win_df.iterrows():
            token = row["token"]
            entry = {
                "window_idx": w_idx,
                "mentions": int(row["mentions"]),
                "avg_engagement": float(row["avg_engagement"]),
                "unique_authors": int(row["unique_authors"]),
                "author_entropy": float(row["author_entropy"]),
                "window_start": row["window_start"],
            }
            token_history.setdefault(token, []).append(entry)

    # For each token, compute metrics for each of its windows
    for token, hist in token_history.items():
        # hist is list of dicts in chronological order
        for idx, cur in enumerate(hist):
            # gather baseline windows (previous BASELINE_WINDOWS windows)
            start_idx = max(0, idx - BASELINE_WINDOWS)
            baseline_entries = hist[start_idx:idx]  # excludes current
            # baseline mentions list
            baseline_mentions = [e["mentions"] for e in baseline_entries if "mentions" in e]
            baseline_engs = [e["avg_engagement"] for e in baseline_entries if "avg_engagement" in e]

            current_mentions = cur["mentions"]
            current_avg_eng = cur["avg_engagement"]
            unique_authors = cur["unique_authors"]
            author_entropy = cur["author_entropy"]

            # baseline medians (safe)
            baseline_mentions_med = float(np.median(baseline_mentions)) if len(baseline_mentions) > 0 else 0.0
            baseline_eng_med = float(np.median(baseline_engs)) if len(baseline_engs) > 0 else 0.0

            # Stable Mention Growth Ratio (MGR): use +1 smoothing to avoid huge blowups
            MGR = (current_mentions + 1.0) / (baseline_mentions_med + 1.0) - 1.0

            # Engagement Velocity (EV): compare avg_engagement now vs baseline engagement median
            if baseline_eng_med > 0:
                EV = current_avg_eng / (baseline_eng_med + EPS)
            else:
                # if no baseline engagement, set neutral EV = 1 (no evidence)
                EV = 1.0

            # Velocity Index (VI): combine growth, uniqueness, engagement (clamp)
            frac_unique = (unique_authors / (current_mentions + EPS)) if current_mentions > 0 else 0.0
            VI = MGR * frac_unique * np.log1p(current_avg_eng)
            # clamp extreme values
            if not np.isfinite(VI):
                VI = 0.0
            VI = max(min(VI, MAX_RATIO), -MAX_RATIO)

            # Normalize VI and EV for combining into CVRS using logistic-like mapping
            def norm01(x):
                try:
                    return 1.0 / (1.0 + np.exp(-0.01 * x))  # gentle squash
                except Exception:
                    return 0.0

            VI_norm = norm01(VI)
            EV_norm = 1.0 / (1.0 + np.exp(-0.01 * (EV - 1.0)))  # center around EV=1

            # Author entropy normalization (map 0..log2(authors) -> 0..1 roughly)
            auth_entropy_score = np.tanh(author_entropy / 4.0)  # heuristic

            # Composite Viral Reliability Score (weights tunable)
            w_vi, w_ev, w_entropy, w_auth = 0.30, 0.30, 0.20, 0.20
            CVRS = w_vi * VI_norm + w_ev * EV_norm + w_entropy * auth_entropy_score + w_auth * (min(unique_authors, 50) / 50.0)

            # Labeling
            label = "LOW"
            if CVRS >= 0.85:
                label = "HOT"
            elif CVRS >= 0.65:
                label = "WATCH"

            results.append({
                "token": token,
                "window_start": cur["window_start"].isoformat(),
                "mentions": current_mentions,
                "unique_authors": unique_authors,
                "avg_engagement": current_avg_eng,
                "mention_growth_ratio": round(float(MGR), 6),
                "velocity_index": round(float(VI), 6),
                "engagement_velocity": round(float(EV), 6),
                "author_entropy": round(float(author_entropy), 6),
                "CVRS": round(float(CVRS), 6),
                "label": label
            })

    if not results:
        raise ValueError("No metrics computed (no windows / tokens).")
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DATA_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    df = load_data(args.input)
    metrics_df = compute_metrics(df)
    metrics_df.to_json(args.output, orient="records", lines=True, force_ascii=False)
    print(f"[DONE] Saved metrics to {args.output} | Tokens analyzed: {metrics_df['token'].nunique()}")

if __name__ == "__main__":
    main()
