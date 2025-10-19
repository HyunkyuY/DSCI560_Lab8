## Reddit Scrape and Doc2Vec Clustering - Setup
```bash
pip install pandas numpy gensim praw python-dotenv matplotlib scikit-learn
```

Create a `.env` file in the same folder:
```
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_app_name
```

## Run
Scrape Reddit + run Doc2Vec clustering:
```bash
python Lab8_Doc2Vec.py --subreddit technology --out technology_posts.csv --k 8 --tsne
```

Use existing CSV (skip scraping):
```bash
python Lab8_Doc2Vec.py --no-scrape --data technology_posts.csv --k 8 --tsne
```

## Output
- `technology_posts.csv`: scraped posts  
- `figs/*.png`: PCA / t-SNE visualizations  
- Terminal: silhouette scores + cluster keywords
