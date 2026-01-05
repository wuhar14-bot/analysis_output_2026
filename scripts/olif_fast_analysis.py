"""
OLIF Citation Analysis - FAST VERSION
======================================
Optimized for speed using vectorized operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from factor_analyzer import FactorAnalyzer
import os
import warnings
from collections import defaultdict
import re

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']

# =============================================================================
# CONFIG
# =============================================================================

WORKING_DIR = r"E:\claude-code\Zhao"
INPUT_FILE = "scopus_export_Jan 4-2026_5012fa77-862f-4048-89a0-ed9493942a56.csv"
OUTPUT_DIR = os.path.join(WORKING_DIR, "analysis_output_2026")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(WORKING_DIR)

COLUMN_MAP = {
    '文獻標題': 'Title',
    '年份': 'Year',
    '機構': 'Affiliations',
    '作者.1': 'Author Keywords',
    '索引關鍵字': 'Index Keywords',
    '來源出版物名稱': 'Source title',
    '連結': 'Link',
    '參考文獻': 'References',
    '原始文獻語言': 'Language',
    '文獻類型': 'Document Type',
    '摘要': 'Abstract',
    '作者': 'Authors',
    '被引用文獻': 'Cited by'
}

BANNED_TITLES = {
    "spondylolisthesis", "abdomen", "adjacent segment disease",
    "adult spinal deformity", "anterior lumbar interbody fusion",
    "lateral lumbar interbody fusion", "lumbar interbody fusion",
    "lumbar spinal stenosis", "introduction", "lumbar lordosis"
}

THRESHOLD = 0.3

# =============================================================================
# FAST MATCHING USING HASH INDEX
# =============================================================================

def fast_citation_matching(df):
    """Fast citation matching using inverted index."""
    print("\n[STEP] Fast citation matching...")

    # Prepare data
    df = df.copy()
    df['Title'] = df['Title'].str.lower().str.strip()
    df['References'] = df['References'].str.lower()

    # Get unique titles
    titles = df['Title'].dropna().unique()
    print(f"  Titles to match: {len(titles)}")

    # Build inverted index: word -> set of titles containing that word
    # This allows O(1) lookup instead of O(n) string search
    title_words = {}
    for title in titles:
        if pd.isna(title) or len(title) < 15:
            continue
        # Use first 5 significant words as key
        words = [w for w in re.split(r'\W+', title) if len(w) > 3][:5]
        for word in words:
            if word not in title_words:
                title_words[word] = set()
            title_words[word].add(title)

    print(f"  Index built with {len(title_words)} unique words")

    # Match references
    pairs = []

    for idx, row in df.iterrows():
        if pd.isna(row['References']):
            continue

        citing_title = row['Title']
        refs = str(row['References'])

        # Find candidate matches using word index
        candidates = set()
        ref_words = [w for w in re.split(r'\W+', refs) if len(w) > 3]

        for word in ref_words:
            if word in title_words:
                candidates.update(title_words[word])

        # Verify matches
        for candidate in candidates:
            if candidate != citing_title and candidate in refs:
                pairs.append({
                    'Title': citing_title,
                    'verified_title': candidate,
                    'Year': row.get('Year'),
                    'Document Type': row.get('Document Type'),
                    'Affiliations': row.get('Affiliations'),
                    'Author Keywords': row.get('Author Keywords'),
                    'Index Keywords': row.get('Index Keywords'),
                    'Source title': row.get('Source title'),
                    'Link': row.get('Link')
                })

    df_pairs = pd.DataFrame(pairs)

    if len(df_pairs) > 0:
        df_pairs = df_pairs.drop_duplicates(subset=['Title', 'verified_title'])

    print(f"  Matched pairs: {len(df_pairs)}")

    return df_pairs


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 60)
    print("OLIF CITATION ANALYSIS 2008-2026 (FAST)")
    print("=" * 60)

    # Load data
    print("\n[STEP] Loading data...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    df = df.rename(columns=COLUMN_MAP)
    print(f"  Loaded: {len(df)} documents")

    # Filter
    print("\n[STEP] Filtering...")
    if 'Language' in df.columns:
        df = df[df['Language'].str.lower().str.contains('english', na=False)]
    print(f"  English docs: {len(df)}")

    if 'Document Type' in df.columns:
        valid = ['article', 'review']
        df = df[df['Document Type'].str.lower().isin(valid)]
    print(f"  Articles/Reviews: {len(df)}")

    df = df[df['References'].notna()]
    print(f"  With references: {len(df)}")

    # Fast matching
    df_pairs = fast_citation_matching(df)

    if len(df_pairs) == 0:
        print("\nERROR: No citation pairs found!")
        return

    # Remove banned titles
    df_pairs = df_pairs[~df_pairs['verified_title'].isin(BANNED_TITLES)]
    print(f"  After removing generic: {len(df_pairs)}")

    # Build dictionary
    print("\n[STEP] Building title dictionary...")
    citing_df = df_pairs[['Title', 'Year', 'Document Type', 'Affiliations',
                          'Author Keywords', 'Index Keywords', 'Source title', 'Link']].drop_duplicates()

    all_titles = list(citing_df['Title'].unique())
    cited_only = [t for t in df_pairs['verified_title'].unique() if t not in all_titles]
    all_titles.extend(cited_only)

    title_to_idx = {t: i for i, t in enumerate(all_titles)}
    n_citing = len(citing_df)

    print(f"  Citing papers: {n_citing}")
    print(f"  Total titles: {len(all_titles)}")

    # Build adjacency matrix
    print("\n[STEP] Building adjacency matrix...")
    df_pairs['citing_idx'] = df_pairs['Title'].map(title_to_idx)
    df_pairs['cited_idx'] = df_pairs['verified_title'].map(title_to_idx)

    n = len(all_titles)
    adj = np.zeros((n, n), dtype=np.int8)

    for _, row in df_pairs.iterrows():
        i, j = int(row['citing_idx']), int(row['cited_idx'])
        adj[i, j] = 1

    print(f"  Matrix shape: {adj.shape}")

    # Bibliographic coupling
    print("\n[STEP] Computing bibliographic coupling...")
    bc = adj[:n_citing, :].dot(adj[:n_citing, :].T)
    print(f"  BC matrix: {bc.shape}")

    # Cosine similarity
    sim = cosine_similarity(bc)
    print(f"  Similarity matrix: {sim.shape}")

    # Factor analysis
    print("\n[STEP] Factor analysis...")
    fa_init = FactorAnalyzer(n_factors=min(n_citing, 50), rotation=None)
    fa_init.fit(sim)
    eigenvalues = fa_init.get_eigenvalues()[0]

    n_factors = np.count_nonzero(eigenvalues > 1)
    n_factors = max(2, min(n_factors, 20))
    print(f"  Factors (eigenvalue > 1): {n_factors}")

    # Scree plot
    plt.figure(figsize=(10, 6))
    x = range(1, min(30, len(eigenvalues)) + 1)
    plt.scatter(x, eigenvalues[:len(x)])
    plt.plot(x, eigenvalues[:len(x)])
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Scree Plot (2008-2026)')
    plt.xlabel('Factor')
    plt.ylabel('Eigenvalue')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'scree_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Final factor analysis
    print(f"\n[STEP] Running factor analysis with {n_factors} factors...")
    fa = FactorAnalyzer(n_factors=n_factors, rotation='promax')
    fa.fit(sim)
    loadings = pd.DataFrame(fa.loadings_, columns=[f'F{i+1}' for i in range(n_factors)])
    loadings.to_csv(os.path.join(OUTPUT_DIR, 'factor_loadings.csv'))

    # Extract streams
    print("\n[STEP] Extracting research streams...")
    print("-" * 40)

    for col in loadings.columns:
        mask = loadings[col] > THRESHOLD
        indices = loadings[mask].index.tolist()
        valid_idx = [i for i in indices if i < len(citing_df)]

        if valid_idx:
            papers = citing_df.iloc[valid_idx]
            papers.to_csv(os.path.join(OUTPUT_DIR, f'{col}.csv'),
                         index=False, encoding='utf-8')

        print(f"  {col}: {len(valid_idx)} papers")

    # Overlap matrix
    binary = (loadings > THRESHOLD).astype(int)
    overlap = binary.T.dot(binary)
    overlap.to_csv(os.path.join(OUTPUT_DIR, 'factor_overlap.csv'))

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total documents: {len(df)}")
    print(f"  Citation pairs: {len(df_pairs)}")
    print(f"  Citing papers analyzed: {n_citing}")
    print(f"  Research factors: {n_factors}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")

    # Top eigenvalues
    print("\nTop 5 eigenvalues:")
    for i, ev in enumerate(eigenvalues[:5]):
        print(f"  Factor {i+1}: {ev:.2f}")

    return {
        'df': df,
        'pairs': df_pairs,
        'loadings': loadings,
        'citing_df': citing_df
    }


if __name__ == "__main__":
    results = main()
