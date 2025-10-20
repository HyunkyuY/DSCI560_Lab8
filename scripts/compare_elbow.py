"""compare_elbow.py

計算 KMeans 的 inertia (elbow) 與 silhouette score (cosine) 的自動化腳本
針對 Doc2Vec 與 Word2Vec 在 dims 100/200/300 的輸出 CSV：
- result_word2vec/reddit_vector_{d}.csv
- results_doc2vec/reddit_vector_{d}.csv

輸出：
- analysis/elbow_results/{method}_{d}_elbow.png
- analysis/elbow_results/{method}_{d}_silhouette.png
- analysis/elbow_results/{method}_{d}_summary.csv

用法：
    python scripts/compare_elbow.py --kmin 2 --kmax 12

不會修改現有程式檔案。
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


def load_vectors(csv_path):
    df = pd.read_csv(csv_path)
    vec_cols = [c for c in df.columns if c.startswith('v')]
    if len(vec_cols) == 0:
        raise ValueError(f"No vector columns found in {csv_path}")
    X = df[vec_cols].values
    return X, df


def evaluate_k_range(X, kmin, kmax, metric='cosine'):
    ks = list(range(kmin, kmax + 1))
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # silhouette needs at least 2 clusters and less than n_samples
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(X):
            try:
                sil = silhouette_score(X, labels, metric=metric)
            except Exception:
                sil = np.nan
        else:
            sil = np.nan
        silhouettes.append(sil)
    return ks, inertias, silhouettes


def plot_elbow_sil(ks, inertias, silhouettes, out_prefix):
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(ks, inertias, '-o', color='C0')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Inertia (SSE)', color='C0')
    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, '-s', color='C1')
    ax2.set_ylabel('Silhouette (cosine)', color='C1')
    plt.title(f'Elbow & Silhouette')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_elbow_sil.png", dpi=200)
    plt.close()

    # also save separate plots
    plt.figure(figsize=(6,3))
    plt.plot(ks, inertias, '-o', color='C0')
    plt.xlabel('k'); plt.ylabel('Inertia (SSE)')
    plt.title('Elbow (Inertia)')
    plt.tight_layout(); plt.savefig(f"{out_prefix}_elbow.png", dpi=200); plt.close()

    plt.figure(figsize=(6,3))
    plt.plot(ks, silhouettes, '-s', color='C1')
    plt.xlabel('k'); plt.ylabel('Silhouette (cosine)')
    plt.title('Silhouette')
    plt.tight_layout(); plt.savefig(f"{out_prefix}_silhouette.png", dpi=200); plt.close()


def main(args):
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / 'analysis' / 'elbow_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = {
        'word2vec': project_root / 'result_word2vec',
        'doc2vec': project_root / 'results_doc2vec'
    }
    dims = args.dims

    summary_rows = []

    for method, folder in methods.items():
        for d in dims:
            csv_path = folder / f'reddit_vector_{d}.csv'
            if not csv_path.exists():
                print(f"Skipping missing file: {csv_path}")
                continue
            print(f"Processing {csv_path}")
            X, df = load_vectors(csv_path)
            # standardize for KMeans (helps inertia comparability)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            ks, inertias, silhouettes = evaluate_k_range(Xs, args.kmin, args.kmax, metric='cosine')
            out_prefix = out_dir / f"{method}_{d}"
            plot_elbow_sil(ks, inertias, silhouettes, str(out_prefix))

            for k, iw, sil in zip(ks, inertias, silhouettes):
                summary_rows.append({
                    'method': method,
                    'dim': d,
                    'k': k,
                    'inertia': iw,
                    'silhouette': sil
                })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / 'elbow_summary.csv', index=False)
    print('Done. Results in', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmin', type=int, default=2)
    parser.add_argument('--kmax', type=int, default=12)
    parser.add_argument('--dims', nargs='+', type=int, default=[100,200,300])
    args = parser.parse_args()
    main(args)
