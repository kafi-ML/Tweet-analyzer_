# Twikit Scraper
Collects latest tweets and metadata from public Twitter/X accounts using Twikit and cookies.

## Run
Single user:
    python main.py --user elonmusk --count 5

Multiple users:
    python main.py --accounts configs/accounts.txt --count 10

Outputs:
    data/raw_tweets.jsonl
    data/parsed_tweets.jsonl
    data/accounts.jsonl
Logs:
    logs/twikit_collector.log
