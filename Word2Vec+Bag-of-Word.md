# README – Lab 8: Word2Vec + Bag-of-Words Embedding

## Overview  
This project implements the **Word2Vec + Bag-of-Words (BoW)** embedding method for Reddit posts as part of **Lab 8 (Part 2)**.  
It converts text posts into fixed-length document vectors by clustering word embeddings into semantic bins and representing each post as a normalized frequency vector over these bins.

The output is later used in **Part 3 (Comparative Analysis)** to compare with the **Doc2Vec** results.

---

## Methodology  
1. **Train Word2Vec model** on Reddit post text (`title + selftext`)  
2. **Cluster word embeddings** into *K* bins using **KMeans** (semantic grouping of similar words)  
3. **Convert each post** into a *K*-dimensional vector based on word-bin frequencies (normalized by post length)  
4. **Cluster document vectors** and evaluate with **silhouette score**  
5. **Visualize** results using **PCA (2D)** plots  

This approach approximates **cosine distance** by **L2-normalizing** the word vectors before clustering.

---

## Requirements  
Make sure you have the following Python packages installed:  

```bash
pip install pandas numpy gensim scikit-learn matplotlib nltk
```

---

## Input Data  
Input file: `technology_posts.csv`  
Required columns:
```
id, title, selftext, subreddit
```
The script automatically combines `title` and `selftext` as the text input for training.

---

## Usage  

### Default Run
```bash
python word2vec_bow.py
```

### Custom Run Example
```bash
python word2vec_bow.py --data technology_posts.csv --out results_word2vec --clusters 5
```

**Arguments:**
| Argument | Description | Default |
|-----------|--------------|----------|
| `--data` | Path to CSV input file | `technology_posts.csv` |
| `--out` | Output directory | `result_word2vec` |
| `--clusters` | Number of document clusters | `5` |

---

## Output Files  

After running, the output folder (e.g. `results_word2vec/`) will contain:

```
results_word2vec/
├── reddit_vector_100_detailed.csv
├── reddit_vector_100.csv
├── reddit_vector_200_detailed.csv
├── reddit_vector_200.csv
├── reddit_vector_300_detailed.csv
├── reddit_vector_300.csv
├── pca_100.png
├── pca_200.png
├── pca_300.png
└── metrics_summary.csv

```

**File descriptions:**

| File | Description |
|------|--------------|
| **`reddit_vector_[dim]_detailed.csv`** | Contains document IDs, text, and their assigned cluster labels. This version is mainly for qualitative inspection — useful to check what kind of posts fall into the same cluster.                                                           |
| **`reddit_vector_[dim].csv`**   | Contains only numerical embeddings and cluster labels. Each row represents one document, with columns `v0 ... vN` corresponding to semantic bin frequencies. This version is used for quantitative analysis or comparison with other models. |
| **`pca_[dim].png`**                    | 2D PCA visualization showing the distribution of clustered document embeddings for each tested dimension. |
| **`metrics_summary.csv`**              | Summary of silhouette scores for each embedding dimension. |

---

## Experiment Configuration  
- **Embedding Dimensions:** 100, 200, 300 (aligned with Doc2Vec configurations)  
- **Clustering Metric:** Cosine distance (via normalization)  
- **Word2Vec Parameters:**  
  - `window=5`, `min_count=3`, `sg=1` (Skip-gram)  
  - `epochs=20`  
- **Document Clustering:** KMeans with `n_clusters=5`

---

## Example Output Log
```
Loaded dataset: technology_posts.csv | 3210 records.
Cleaned dataset size: 3187
Tokenizing text...

=== Training Word2Vec with dim=100 ===
dim=100 | silhouette=0.3285
=== Training Word2Vec with dim=200 ===
dim=200 | silhouette=0.3412
=== Training Word2Vec with dim=300 ===
dim=300 | silhouette=0.3569

Experiment complete! Results saved to: results_word2vec
```

---

## Notes  
- This script uses **cosine-style distance** by normalizing word embeddings before clustering.  
- PCA is for **visualization only** and does not affect clustering results.  
- Results are meant to be compared directly with **Doc2Vec** outputs under the same vector dimensions.
