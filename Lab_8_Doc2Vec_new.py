import argparse, sys, random
from pathlib import Path
from collections import Counter
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def parse_args():
    p = argparse.ArgumentParser(description="Doc2Vec + KMeans on Reddit posts")
    p.add_argument("--data", type=str, default="technology_posts.csv",
                   help="Path to the fixed Reddit posts CSV")
    p.add_argument("--k", type=int, default=8, help="Number of clusters for KMeans")
    p.add_argument("--tsne", action="store_true", help="Also compute t-SNE plots (slower)")
    p.add_argument("--out", type=str, default="results.csv",
                   help="CSV to save key metrics (one row per configuration)")
    p.add_argument("--out_dir", type=str, default="results_doc2vec",
                   help="Directory to save per-dimension vectors (reddit_vector_*.csv), plots, and metrics_summary.csv")
    return p.parse_args()


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

    # ensure nltk resources available (quiet)
    try:
        nltk.data.find('tokenizers/punkt')
    except Exception:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except Exception:
        nltk.download('stopwords', quiet=True)
    # use NLTK english stopwords to match word2vec preprocessing
    stop_words = set(nltk_stopwords.words('english'))

    def tokenize(txt):
        # remove non-letters, lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', str(txt).lower())
        # nltk tokenize, filter stopwords and short tokens
        tokens = [t for t in nltk.word_tokenize(text) if t not in stop_words and len(t) > 2]
        return tokens

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
    mapping = {}
    clusters = sorted(set(labels.tolist()))
    # try to use nltk stopwords if available (keeps preprocessing consistent with word2vec)
    try:
        sw = set(nltk.corpus.stopwords.words('english'))
    except Exception:
        sw = STOPWORDS
    for c in clusters:
        idx = np.where(labels == c)[0]
        counts = Counter()
        for i in idx:
            toks = tokens_list[i]
            filtered = [t for t in toks if t.isalpha() and len(t) > 2 and t not in sw]
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
        print(f"Training {name}...")
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

    cmap = plt.get_cmap("tab10")
    colors = [cmap(int(c) % 10) for c in labels]

    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=colors, s=8)
    plt.title(title)
    plt.xlabel("PCA Component 1" if method == "pca" else "t-SNE dim 1")
    plt.ylabel("PCA Component 2" if method == "pca" else "t-SNE dim 2")

    if cluster_keywords is not None:
        handles, legend_labels = [], []
        for c in sorted(set(labels.tolist())):
            color = cmap(int(c) % 10)
            terms = cluster_keywords.get(c, [])
            text = f"{c}: " + (", ".join(terms) if terms else "(no terms)")
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
            legend_labels.append(text)
        plt.legend(handles, legend_labels, title="Clusters (top terms)", loc="best", fontsize="small")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.show()
    plt.close()


def embeddings_and_clusters(csv_path: Path, k: int, do_tsne: bool, out_csv: Path, out_dir: Path):
    df = load_and_tokenize(csv_path)
    # Prepare IDs (use 'id' column if present, else row indices)
    ids = df["id"].astype(str).tolist() if "id" in df.columns else [str(i) for i in range(len(df))]
    # Prepare output dir (vectors/plots/metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = tagged_docs(df["tokens"])
    n_posts = len(df)
    print(f"Dataset: {n_posts} posts\n")

    models = train_models(docs)
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(exist_ok=True)

    metrics_rows = []

    for name, model in models.items():
        print(f"\nRunning {name}...")
        X = infer_embeddings(model, docs)
        labels, sil, km = kmeans_cosine(X, k)
        # --- Save per-document vectors & clusters in a unified schema (id, cluster, v0...v{dim-1})
        dim = model.vector_size
        vec_cols = [f"v{i}" for i in range(dim)]
        out_df = pd.DataFrame(X, columns=vec_cols)
        out_df.insert(0, "cluster", labels)
        out_df.insert(0, "id", ids)
        csv_path = out_dir / f"reddit_vector_{dim}.csv"
        out_df.to_csv(csv_path, index=False)
        print(f"Saved vectors: {csv_path}")

        cluster_counts = np.bincount(labels)

        print(f"Silhouette (cosine): {sil:.4f}")
        print("Cluster sizes:", cluster_counts.tolist())

        cluster_keywords = cluster_top_keywords(df["tokens"].tolist(), labels, topn=5)
        plot_2d(X, labels, method="pca",
                title=f"Doc2Vec {dim}-dim Clusters, silhouette={sil:.3f}",
                save_path=fig_dir / f"pca_{dim}.png",
                cluster_keywords=cluster_keywords)
        if do_tsne:
            plot_2d(X, labels, method="tsne",
                    title=f"Doc2Vec {dim}-dim t-SNE (K={k})",
                    save_path=fig_dir / f"tsne_{dim}.png",
                    cluster_keywords=cluster_keywords)
        metrics_rows.append({
            "Method": "Doc2Vec",
            "Dim": dim,
            "Config": name,
            "K": k,
            "NumPosts": n_posts,
            "Silhouette": round(sil, 6),
            "ClusterSizes": ",".join(map(str, cluster_counts.tolist()))
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nKey metrics saved to {out_csv}")
    # Also keep a consolidated metrics file under out_dir for Part 3
    ms_path = out_dir / "metrics_summary.csv"
    if ms_path.exists():
        old = pd.read_csv(ms_path)
        metrics_df = pd.concat([old, metrics_df], ignore_index=True)
    metrics_df.to_csv(ms_path, index=False)
    print(f"Metrics summary updated: {ms_path}")

def main():
    args = parse_args()
    csv_path = Path(args.data)
    embeddings_and_clusters(csv_path, k=args.k, do_tsne=args.tsne, out_csv=Path(args.out), out_dir=Path(args.out_dir))

if __name__ == "__main__":
    main()
