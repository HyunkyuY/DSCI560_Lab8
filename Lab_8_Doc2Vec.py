import argparse, os, sys, time, random
from pathlib import Path
from collections import Counter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import praw
from praw.models import Submission
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def parse_args():
    p = argparse.ArgumentParser(description="Reddit scrape + Doc2Vec + clustering")
    p.add_argument("--subreddit", type=str, default="technology", help="Subreddit to scrape")
    p.add_argument("--sort", choices=["new", "hot", "top", "rising"], default="new", help="Reddit listing to pull")
    p.add_argument("--time-filter", choices=["all","year","month","week","day","hour"], default="year",
                   help="Only used when sort=top")
    p.add_argument("--out", type=str, default="technology_posts.csv", help="CSV output path for scraped posts")
    p.add_argument("--append", action="store_true", help="Append to existing CSV (dedup by id)")

    p.add_argument("--no-scrape", action="store_true", help="Skip scraping; use --data CSV instead")
    p.add_argument("--data", type=str, help="Path to an existing CSV; used with --no-scrape")

    # Embedding/Clustering
    p.add_argument("--k", type=int, default=8, help="Number of clusters for KMeans")
    p.add_argument("--per-cluster", type=int, default=3, help="(kept for compatibility; not saving samples)")
    p.add_argument("--tsne", action="store_true", help="Also compute t-SNE plots (slower)")
    return p.parse_args()


def init_reddit():
    load_dotenv()
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "doc2vec-scraper")
    if not all([client_id, client_secret, user_agent]):
        raise RuntimeError("Missing Reddit API credentials in .env (REDDIT_CLIENT_ID/SECRET/USER_AGENT)")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        ratelimit_seconds=5,
    )


def as_row(s: Submission):
    return {
        "id": s.id,
        "created_utc": int(getattr(s, "created_utc", 0) or 0),
        "author": str(getattr(getattr(s, "author", None), "name", "") or ""),
        "title": s.title or "",
        "selftext": (s.selftext or "").replace("\x00", " "),
        "score": int(getattr(s, "score", 0) or 0),
        "num_comments": int(getattr(s, "num_comments", 0) or 0),
        "upvote_ratio": float(getattr(s, "upvote_ratio", 0.0) or 0.0),
        "over_18": bool(getattr(s, "over_18", False)),
        "spoiler": bool(getattr(s, "spoiler", False)),
        "stickied": bool(getattr(s, "stickied", False)),
        "link_flair_text": s.link_flair_text or "",
        "permalink": f"https://reddit.com{s.permalink}" if getattr(s, "permalink", None) else "",
        "url": s.url or "",
        "subreddit": str(getattr(s.subreddit, "display_name", "")),
        "distinguished": s.distinguished or "",
    }


def fetch_listing(sr, sort, time_filter):
    if sort == "new":
        gen = sr.new(limit=None)
    elif sort == "hot":
        gen = sr.hot(limit=None)
    elif sort == "rising":
        gen = sr.rising(limit=None)
    else:
        gen = sr.top(time_filter=time_filter, limit=None)
    for s in gen:
        yield s


def scrape_to_csv(subreddit: str, sort: str, time_filter: str, out_path: Path, append: bool):
    reddit = init_reddit()
    sr = reddit.subreddit(subreddit)

    if append and out_path.exists():
        df_existing = pd.read_csv(out_path)
        seen = set(df_existing["id"].astype(str).tolist())
    else:
        df_existing = None
        seen = set()

    rows, grabbed = [], 0
    start = time.time()
    try:
        for s in fetch_listing(sr, sort, time_filter):
            if s is None:
                continue
            sid = getattr(s, "id", None)
            if not sid or sid in seen:
                continue
            rows.append(as_row(s))
            seen.add(sid)
            grabbed += 1
            if grabbed % 200 == 0:
                print(f"Scraped {grabbed} posts...")
                pd.DataFrame(rows).to_csv(out_path, index=False)
        pd.DataFrame(rows).to_csv(out_path, index=False)

        if df_existing is not None:
            df_new = pd.read_csv(out_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
            df_all = df_all.drop_duplicates(subset=["id"]).sort_values("created_utc", ascending=False)
            df_all.to_csv(out_path, index=False)

        elapsed = time.time() - start
        print(f"Scraping complete. {grabbed} total posts saved to {out_path} (in {elapsed:.1f}s).")
    except KeyboardInterrupt:
        print("Scraping stopped manually. Writing partial results...")
        if rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)


def load_and_tokenize(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        print(f"Error: dataset not found at {csv_path}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(csv_path)

    def join_text(row):
        parts = []
        if "title" in df.columns:
            parts.append(str(row.get("title", "")))
        for col in ("selftext", "body", "text"):
            if col in df.columns:
                parts.append(str(row.get(col, "")))
        return " ".join(parts)

    df["raw_text"] = df.apply(join_text, axis=1).fillna("").astype(str)
    df = df[df["raw_text"].str.strip().astype(bool)].reset_index(drop=True)

    def tokenize(txt):
        return simple_preprocess(txt, deacc=True, min_len=2, max_len=30)

    df["tokens"] = df["raw_text"].map(tokenize)
    return df

def tagged_docs(tokens_series: pd.Series):
    return [TaggedDocument(words=toks, tags=[str(i)]) for i, toks in enumerate(tokens_series)]


STOPWORDS = {
    "the","a","an","and","or","for","of","to","in","on","with","is","are","was","were","be",
    "this","that","it","as","at","by","from","you","your","i","we","they","them","our","us",
    "if","but","so","not","no","yes","just","like","can","will","would","could","should",
    "amp","https","http","www","com"
}


def cluster_top_keywords(tokens_list, labels, topn=5):
    """
    tokens_list: list[list[str]] from df['tokens']
    labels: np.ndarray of cluster ids
    returns: dict[int, list[str]] mapping cluster_id -> top tokens
    """
    mapping = {}
    clusters = sorted(set(labels.tolist()))
    for c in clusters:
        idx = np.where(labels == c)[0]
        counts = Counter()
        for i in idx:
            toks = tokens_list[i]
            filtered = [t for t in toks if t.isalpha() and len(t) > 2 and t not in STOPWORDS]
            counts.update(filtered)
        mapping[c] = [w for w, _ in counts.most_common(topn)]
    return mapping


def train_models(documents):
    configs = {
        "Configuration A": dict(vector_size=100, min_count=3, window=5, dm=1, workers=4, epochs=20),
        "Configuration B": dict(vector_size=200, min_count=5, window=5, dm=1, workers=4, epochs=40),
        "Configuration C": dict(vector_size=300, min_count=5, window=7, dm=1, workers=4, epochs=60),
    }
    models = {}
    for name, params in configs.items():
        print(f"Training {name} with parameters: {params}")
        build_params = {k: v for k, v in params.items() if k != "epochs"}
        model = Doc2Vec(**build_params)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=params["epochs"])
        models[name] = model
    return models

def infer_embeddings(model, docs, epochs=30):
    return np.vstack([model.infer_vector(td.words, epochs=epochs) for td in docs])

def kmeans_cosine(X, k=8):
    Xn = normalize(X)
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(Xn)
    sil = silhouette_score(Xn, labels, metric="cosine")
    return labels, float(sil), km


def plot_2d(X, labels, method="pca", title="", save_path=None, cluster_keywords=None):
    if method == "pca":
        Z = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    elif method == "tsne":
        Z = TSNE(n_components=2, random_state=RANDOM_STATE, init="random", perplexity=30).fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    cmap = plt.cm.get_cmap("tab10")
    point_colors = [cmap(int(c) % 10) for c in labels]

    plt.figure(figsize=(7,5))
    plt.scatter(Z[:,0], Z[:,1], c=point_colors, s=8)
    plt.title(title)
    plt.xlabel("PCA Component 1" if method=="pca" else "t-SNE dim 1")
    plt.ylabel("PCA Component 2" if method=="pca" else "t-SNE dim 2")

    handles, legend_labels = [], []
    for c in sorted(set(labels.tolist())):
        color = cmap(int(c) % 10)
        terms = (cluster_keywords or {}).get(c, [])
        text = f"{c}: " + (", ".join(terms) if terms else "(no terms)")
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(text)

    plt.legend(handles, legend_labels, title="Clusters (top terms)", loc="best", fontsize="small")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()
    plt.close()


def embeddings_and_clusters(csv_path: Path, k: int, per_cluster: int, do_tsne: bool):
    df = load_and_tokenize(csv_path)
    docs = tagged_docs(df["tokens"])
    print(f"{len(df)} posts after preprocessing.\n")

    models = train_models(docs)
    fig_dir = Path("./figs")
    fig_dir.mkdir(exist_ok=True)

    results = {}

    for name, model in models.items():
        print(f"\nRunning {name}...")
        X = infer_embeddings(model, docs, epochs=30)
        labels, sil, km = kmeans_cosine(X, k=k)
        results[name] = sil
        print(f"Silhouette (cosine) = {sil:.4f}")

        cluster_counts = np.bincount(labels)
        print("Cluster sizes:", cluster_counts.tolist())

        cluster_keywords = cluster_top_keywords(df["tokens"].tolist(), labels, topn=5)
        print("Top terms per cluster:")
        for c in sorted(cluster_keywords.keys()):
            print(f"  {c}: {', '.join(cluster_keywords[c])}")

        plot_2d(
            X, labels, method="pca",
            title=f"{name} PCA (K={k})",
            save_path=fig_dir / f"{name}_pca.png",
            cluster_keywords=cluster_keywords
        )
        if do_tsne:
            plot_2d(
                X, labels, method="tsne",
                title=f"{name} t-SNE (K={k})",
                save_path=fig_dir / f"{name}_tsne.png",
                cluster_keywords=cluster_keywords
            )

    print("\nSilhouette Scores Summary:")
    for name, score in results.items():
        print(f"{name}: {score:.4f}")


def main():
    args = parse_args()

    if not args.no_scrape:
        out_path = Path(args.out)
        scrape_to_csv(args.subreddit, args.sort, args.time_filter, out_path, args.append)
        csv_path = out_path
    else:
        if not args.data:
            print("Error: --no-scrape requires --data <csv>", file=sys.stderr)
            sys.exit(2)
        csv_path = Path(args.data)

    embeddings_and_clusters(csv_path, k=args.k, per_cluster=args.per_cluster, do_tsne=args.tsne)

if __name__ == "__main__":
    main()
