# ViralToken – AI-Powered Memecoin Trend Analyzer (In Progress)

ViralToken is a Python-based analytics pipeline that tracks **emerging memecoin trends** from **verified crypto influencer tweets** in real time.  
It combines tweet scraping, token extraction, and multi-metric analytics to detect viral coins early — all without needing the official Twitter API.

---

## 🚀 Features (Current Progress)

✅ **Data Scraping (Twikit)**  
- Fetches tweets from verified influencers using the [Twikit](https://github.com/d60/twikit) client.  
- Extracts full tweet objects — text, timestamps, likes, retweets, replies, and media links.

✅ **Token Extraction (NLP)**  
- Cleans and preprocesses tweet text.  
- Identifies potential crypto tickers using rule-based parsing + NLP (spacy).  
- Removes noise such as “ALL”, “CAN”, “SEE”, etc. for high precision.

✅ **Analytics & Metrics (Stage 1)**  
- Calculates early-stage social metrics:
  - Mentions count & growth ratio  
  - Unique author ratio  
  - Weighted engagement velocity  
  - Author entropy (diversity metric)
- Outputs token-level statistics to JSONL for further modeling.

---

## 🧠 Tech Stack

- **Language:** Python 3.10+  
- **Libraries:**  
  - `twikit` — for scraping X (Twitter) data  
  - `pandas`, `numpy` — for data aggregation  
  - `spacy` — for NLP-based token extraction  
  - `argparse`, `jsonlines`, `logging` — for CLI-based control  

---

## 🗂️ Project Structure

