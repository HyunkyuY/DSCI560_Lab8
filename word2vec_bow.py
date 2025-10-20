import argparse
import pandas as pd
import numpy as np
import os
import re
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords

# Preprocessing
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = [t for t in nltk.word_tokenize(text) if t not in stop_words and len(t) > 2]
    return tokens


# Main embedding + clustering pipeline
def train_word2vec_bow(data_path, output_dir, dims=[100, 200, 300], n_clusters=5):
    # Load data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV file not found: {data_path}.")
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} | {len(df)} records.")
    
    # Combine title + selftext
    df['text'] = (df['title'].fillna('') + df['selftext'].fillna('')).str.strip()
    
    # Drop empty texts
    df = df[df['text'].str.strip().astype(bool)]
    print(f"Cleaned dataset size: {len(df)}")
    
    # Tokenization
    print("Tokenizing text...")
    tokenized_docs = [preprocess(t) for t in df['text'].astype(str).tolist()]
    os.makedirs(output_dir, exist_ok=True)
    
    results = []

    
    # For each embedding dimension
    for dim in dims:
        print(f"\n=== Training Word2Vec with dim={dim} ===")
        model = Word2Vec(sentences=tokenized_docs, vector_size=dim, window=5, min_count=3, sg=1, epochs=20, seed=42)
    
        # Build bins (word clusters)
        X = model.wv.vectors
        X = normalize(X)  # L2 normalization which uses direction instead of length -> for approximating cosine distance
        vocab_size = len(X)
        if vocab_size < dim:
            print(f"Not enough vocab ({vocab_size}) for {dim} bins. Skipping this dimension.")
            continue
        
        km = KMeans(n_clusters=dim, random_state=42, n_init=10)
        km.fit(X)
        word_to_bin = {w: km.labels_[i] for i, w in enumerate(model.wv.index_to_key)}
        
        # Create document vectors (normalized bin frequencies)
        def doc_vec(tokens):
            vec = np.zeros(dim)
            valid = [t for t in tokens if t in word_to_bin]
            for t in valid:
                vec[word_to_bin[t]] += 1
            if len(valid) > 0:
                vec /= len(valid)
            return vec
        
        vectors = np.array([doc_vec(t) for t in tokenized_docs])
        
        # Document-level clustering
        km_doc = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km_doc.fit_predict(vectors)
        score = silhouette_score(vectors, labels)
        results.append({'dimension': dim, 'silhouette': score})
        print(f"dim={dim} | silhouette={score:.4f}")
        
        # Save cluster assignments
        df_out = pd.DataFrame({
            'id': df['id'],
            'text': df.get('text'),
            'cluster': labels
        })
        df_out.to_csv(f"{output_dir}/reddit_vector_{dim}_detailed.csv", index=False)
        
        # Matrix version: full vector representation (for comparison)
        vec_out = pd.DataFrame(vectors, columns=[f'v{i}' for i in range(vectors.shape[1])])
        vec_out.insert(0, 'cluster', labels)
        vec_out.insert(0, 'id', df['id'].values)
        vec_out.to_csv(f"{output_dir}/reddit_vector_{dim}.csv", index=False)
        
        # PCA Visualization
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)
        plt.figure(figsize=(6,5))
        plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='rainbow', s=8)
        plt.title(f"Word2Vec+BOW {dim}-dim Clusters, silhouette={score:.3f}")
        plt.savefig(f"{output_dir}/pca_{dim}.png", dpi=300)
        plt.close()
        
    # Save metrics summary
    pd.DataFrame(results).to_csv(f"{output_dir}/metrics_summary.csv", index=False)
    print("\nExperiment complete! Results saved to:", output_dir)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Word2Vec + BOW embeddings on Reddit posts")
    parser.add_argument("--data", type=str, default="technology_posts.csv", help="Path to CSV data")
    parser.add_argument("--out", type=str, default="result_word2vec", help="Output directory")
    parser.add_argument("--clusters", type=int, default=5, help="Number of document clusters")
    args = parser.parse_args()
    
    train_word2vec_bow(args.data, args.out, n_clusters=args.clusters)
        