"""Fetch article text from URLs in technology_posts.csv and save id + article_text to CSV

Usage:
    python scripts/fetch_articles.py --input technology_posts.csv --output technology_posts_with_article.csv --max 500

This script uses requests + readability + BeautifulSoup for extraction and caches results in cache/articles/<id>.json
"""

import argparse
import requests
from readability import Document
from bs4 import BeautifulSoup
import time
import os
import json
import pandas as pd
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; dsci560-bot/1.0; +mailto:you@example.com)"}
TIMEOUT = 10
RETRY = 2
DELAY = 1.0  # seconds between requests
CACHE_DIR = Path("cache/articles")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_once(url, id_):
    cache_file = CACHE_DIR / f"{id_}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            pass
    for attempt in range(RETRY):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if r.status_code != 200:
                out = {"status": r.status_code, "content": ""}
                cache_file.write_text(json.dumps(out, ensure_ascii=False))
                return out
            html = r.text
            doc = Document(html)
            content_html = doc.summary()
            soup = BeautifulSoup(content_html, "lxml")
            text = soup.get_text(separator="\n").strip()
            out = {"status": 200, "content": text, "title": doc.title()}
            cache_file.write_text(json.dumps(out, ensure_ascii=False))
            time.sleep(DELAY)
            return out
        except Exception as e:
            last_exc = e
            time.sleep(1)
    return {"status": "error", "error": str(last_exc), "content": ""}


def enrich_csv(input_csv, output_csv, url_col="url", id_col="id", max_rows=None):
    df = pd.read_csv(input_csv)
    if max_rows:
        df = df.head(max_rows).copy()
    contents = []
    for i, row in df.iterrows():
        url = row.get(url_col, "")
        id_ = str(row.get(id_col, i))
        if not url or pd.isna(url):
            contents.append({"id": id_, "article_text": "", "status": "no_url"})
            continue
        res = fetch_once(url, id_)
        contents.append({"id": id_, "article_text": res.get("content", ""), "status": res.get("status", "error")})
    cdf = pd.DataFrame(contents)
    cdf["id"] = cdf["id"].astype(str)
    if id_col in df.columns:
        df[id_col] = df[id_col].astype(str)
        merged = df.merge(cdf, left_on=id_col, right_on="id", how="left")
    else:
        merged = pd.concat([df.reset_index(drop=True), cdf.reset_index(drop=True)], axis=1)
    merged.to_csv(output_csv, index=False)
    print("Saved:", output_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='technology_posts.csv')
    parser.add_argument('--output', type=str, default='technology_posts_with_article.csv')
    parser.add_argument('--max', type=int, default=500, help='Max rows to process (for quick tests)')
    args = parser.parse_args()
    enrich_csv(args.input, args.output, max_rows=args.max)
