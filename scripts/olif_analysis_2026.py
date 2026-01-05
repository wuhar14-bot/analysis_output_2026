"""
OLIF Citation Network & Research Streams Analysis (2008-2026)
==============================================================
Updated script for new Scopus export with Chinese column headers.

Author: Hao Wu
Date: January 2026
Data: Scopus export 2008-2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from factor_analyzer import FactorAnalyzer
import os
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

WORKING_DIR = r"E:\claude-code\Zhao"
os.chdir(WORKING_DIR)

# Input file
INPUT_FILE = "scopus_export_Jan 4-2026_5012fa77-862f-4048-89a0-ed9493942a56.csv"

# Output directory
OUTPUT_DIR = os.path.join(WORKING_DIR, "analysis_output_2026")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column name mapping (Traditional Chinese to English)
COLUMN_MAP = {
    '文獻標題': 'Title',
    '年份': 'Year',
    '機構': 'Affiliations',
    '作者.1': 'Author Keywords',  # This might be author keywords
    '索引關鍵字': 'Index Keywords',
    '來源出版物名稱': 'Source title',
    '連結': 'Link',
    '參考文獻': 'References',
    '原始文獻語言': 'Language of Original Document',
    '文獻類型': 'Document Type',
    '摘要': 'Abstract',
    '作者': 'Authors',
    'Author full names': 'Author full names',
    '被引用文獻': 'Cited by'
}

# Analysis parameters
FACTOR_LOADING_THRESHOLD = 0.3
EIGENVALUE_THRESHOLD = 1.0

# Banned generic titles
BANNED_TITLES = [
    "spondylolisthesis", "abdomen", "adjacent segment disease",
    "adult degenerative scoliosis", "adult spinal deformity",
    "anterior lumbar interbody fusion", "degenerative lumbar scoliosis",
    "degenerative scoliosis", "degenerative spondylolisthesis",
    "introduction", "isthmic spondylolisthesis.",
    "lateral lumbar interbody fusion", "lumbar degenerative disk disease",
    "lumbar foraminal stenosis", "lumbar interbody fusion",
    "lumbar lordosis", "lumbar spinal stenosis"
]

# =============================================================================
# STEP 1: LOAD AND STANDARDIZE DATA
# =============================================================================

def load_and_standardize_data():
    """Load Scopus data and standardize column names."""
    print("=" * 60)
    print("STEP 1: Loading and standardizing data")
    print("=" * 60)

    # Load with UTF-8 encoding
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"Total documents loaded: {len(df)}")

    # Rename columns to English
    df = df.rename(columns=COLUMN_MAP)

    # Check available columns
    print(f"\nAvailable columns after mapping:")
    for col in ['Title', 'Year', 'References', 'Document Type', 'Language of Original Document']:
        if col in df.columns:
            print(f"  [OK] {col}")
        else:
            print(f"  [MISSING] {col}")

    # Check References column
    if 'References' in df.columns:
        refs_available = df['References'].notna().sum()
        print(f"\nDocuments with references: {refs_available}")

    return df


# =============================================================================
# STEP 2: FILTER DATA
# =============================================================================

def filter_data(df):
    """Filter to English articles/reviews."""
    print("\n" + "=" * 60)
    print("STEP 2: Filtering data")
    print("=" * 60)

    initial_count = len(df)

    # Filter by language if column exists
    if 'Language of Original Document' in df.columns:
        df_filtered = df[df['Language of Original Document'].str.lower().str.contains('english', na=False)]
        print(f"English documents: {len(df_filtered)}")
    else:
        df_filtered = df.copy()
        print("Language column not found, keeping all documents")

    # Filter by document type if column exists
    if 'Document Type' in df.columns:
        valid_types = ['article', 'review', 'book chapter', 'editorial']
        mask = df_filtered['Document Type'].str.lower().isin(valid_types)
        df_filtered = df_filtered[mask]
        print(f"After type filter (Article/Review): {len(df_filtered)}")
        print("\nDocument type distribution:")
        print(df_filtered['Document Type'].value_counts())

    # Must have references
    if 'References' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['References'].notna()]
        print(f"\nWith references available: {len(df_filtered)}")

    return df_filtered


# =============================================================================
# STEP 3: EXTRACT REFERENCES
# =============================================================================

def extract_references(df):
    """Extract and explode references."""
    print("\n" + "=" * 60)
    print("STEP 3: Extracting references")
    print("=" * 60)

    # Select columns
    cols_to_keep = ['Title', 'Year', 'Affiliations', 'Author Keywords',
                    'Document Type', 'Index Keywords', 'Source title',
                    'Link', 'References']
    cols_available = [c for c in cols_to_keep if c in df.columns]

    base_df = df[cols_available].copy()

    # Lowercase for matching
    base_df['Title'] = base_df['Title'].str.lower()
    base_df['References'] = base_df['References'].str.lower()

    # Explode references
    base_df = base_df.assign(
        References=base_df['References'].str.split(';')
    ).explode('References').reset_index(drop=True)

    # Clean up
    base_df['References'] = base_df['References'].str.strip()

    print(f"Total citation pairs: {len(base_df)}")

    return base_df


# =============================================================================
# STEP 4: BUILD REFERENCE DATABASE FROM CITATIONS
# =============================================================================

def build_reference_database(df_references):
    """Build reference database from extracted citations."""
    print("\n" + "=" * 60)
    print("STEP 4: Building reference database")
    print("=" * 60)

    # Get unique references
    all_refs = df_references['References'].dropna().unique()
    print(f"Unique references found: {len(all_refs)}")

    # Extract titles from reference strings
    # Reference format typically: "author, title, journal, year, vol, page"
    ref_titles = []
    for ref in all_refs:
        if pd.isna(ref) or len(str(ref)) < 10:
            continue
        # Try to extract meaningful part
        ref_clean = str(ref).strip()
        if len(ref_clean) > 5:
            ref_titles.append(ref_clean)

    print(f"Valid references: {len(ref_titles)}")

    return ref_titles


# =============================================================================
# STEP 5: SELF-CITATION MATCHING
# =============================================================================

def match_self_citations(df_references):
    """Match references within the dataset itself."""
    print("\n" + "=" * 60)
    print("STEP 5: Matching citations within dataset")
    print("=" * 60)

    # Get unique titles from our dataset
    our_titles = df_references['Title'].dropna().unique()
    print(f"Titles in our dataset: {len(our_titles)}")

    # Match references to titles in our dataset
    df_pair = pd.DataFrame()

    print("Matching references (this may take a few minutes)...")

    for i, title in enumerate(our_titles):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(our_titles)}")

        if pd.isna(title) or len(str(title)) < 10:
            continue

        # Check if this title appears in any reference
        mask = df_references['References'].str.contains(title, na=False, regex=False)
        if mask.any():
            matches = df_references.loc[mask].copy()
            matches['verified_title'] = title
            df_pair = pd.concat([df_pair, matches], ignore_index=True)

    print(f"\nMatched citation pairs: {len(df_pair)}")

    if len(df_pair) > 0:
        # Remove duplicates
        df_pair = df_pair.drop_duplicates(subset=['Title', 'verified_title'])
        print(f"After deduplication: {len(df_pair)}")
        print(f"Unique citing papers: {len(df_pair['Title'].unique())}")
        print(f"Unique cited papers: {len(df_pair['verified_title'].unique())}")

    return df_pair


# =============================================================================
# STEP 6: REMOVE GENERIC TITLES
# =============================================================================

def remove_generic_titles(df_pair, banned_titles):
    """Remove non-specific reference titles."""
    print("\n" + "=" * 60)
    print("STEP 6: Removing generic titles")
    print("=" * 60)

    if len(df_pair) == 0:
        print("No pairs to filter")
        return df_pair

    initial_count = len(df_pair)
    df_clean = df_pair[~df_pair['verified_title'].isin(banned_titles)]

    print(f"Removed {initial_count - len(df_clean)} pairs with generic titles")
    print(f"Remaining pairs: {len(df_clean)}")

    return df_clean


# =============================================================================
# STEP 7: BUILD TITLE DICTIONARY
# =============================================================================

def build_title_dictionary(df_pair):
    """Create dictionary mapping titles to indices."""
    print("\n" + "=" * 60)
    print("STEP 7: Building title dictionary")
    print("=" * 60)

    if len(df_pair) == 0:
        print("No pairs available")
        return pd.DataFrame(), pd.DataFrame()

    # Get unique citing papers with metadata
    cols = ['Title', 'Year', 'Affiliations', 'Author Keywords',
            'Document Type', 'Index Keywords', 'Source title', 'Link']
    cols_available = [c for c in cols if c in df_pair.columns]

    dictionary_df = df_pair[cols_available].drop_duplicates()

    # Get unique cited papers
    cited_titles = df_pair['verified_title'].unique()
    citing_titles = set(dictionary_df['Title'])
    references_only = [t for t in cited_titles if t not in citing_titles]

    # Combine
    ref_df = pd.DataFrame({'Title': references_only})
    dictionary_full = pd.concat([dictionary_df, ref_df], ignore_index=True)

    print(f"Citing papers: {len(dictionary_df)}")
    print(f"Referenced-only papers: {len(references_only)}")
    print(f"Total unique titles: {len(dictionary_full)}")

    return dictionary_df, dictionary_full


# =============================================================================
# STEP 8: CREATE ADJACENCY MATRIX
# =============================================================================

def create_adjacency_matrix(df_pair, dictionary_full):
    """Create citing-cited adjacency matrix."""
    print("\n" + "=" * 60)
    print("STEP 8: Creating adjacency matrix")
    print("=" * 60)

    if len(df_pair) == 0 or len(dictionary_full) == 0:
        print("Insufficient data for matrix")
        return None, None, None

    # Create title-to-index mapping
    title_index = {title: i for i, title in enumerate(dictionary_full['Title'])}

    # Map titles to indices
    df_indexed = df_pair.copy()
    df_indexed['Title_index'] = df_indexed['Title'].map(title_index)
    df_indexed['Reference_Index'] = df_indexed['verified_title'].map(title_index)

    # Remove unmapped
    df_indexed = df_indexed.dropna(subset=['Title_index', 'Reference_Index'])
    df_indexed['Title_index'] = df_indexed['Title_index'].astype(int)
    df_indexed['Reference_Index'] = df_indexed['Reference_Index'].astype(int)

    # Get unique indices
    title_idx = df_indexed['Title_index'].unique()
    ref_idx = df_indexed['Reference_Index'].unique()
    idx_union = np.union1d(title_idx, ref_idx)

    # Create cross-tabulation matrix
    adj_matrix = pd.crosstab(df_indexed['Title_index'], df_indexed['Reference_Index'])
    adj_matrix = adj_matrix.reindex(index=idx_union, columns=idx_union, fill_value=0)
    adj_matrix = adj_matrix.to_numpy()

    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    return adj_matrix, df_indexed, title_index


# =============================================================================
# STEP 9: BIBLIOGRAPHIC COUPLING
# =============================================================================

def calculate_similarity_matrix(adj_matrix, n_citing_papers):
    """Calculate bibliographic coupling with cosine similarity."""
    print("\n" + "=" * 60)
    print("STEP 9: Calculating similarity matrix")
    print("=" * 60)

    if adj_matrix is None:
        return None, None

    # Bibliographic coupling
    bc_matrix = adj_matrix.dot(adj_matrix.transpose())
    print(f"Bibliographic coupling matrix: {bc_matrix.shape}")

    # Subset for citing papers
    n = min(n_citing_papers, bc_matrix.shape[0])
    bc_subset = bc_matrix[:n, :n]
    print(f"Subset for analysis: {bc_subset.shape}")

    # Cosine similarity
    similarity_matrix = cosine_similarity(bc_subset)

    return similarity_matrix, bc_matrix


# =============================================================================
# STEP 10: FACTOR ANALYSIS
# =============================================================================

def determine_n_factors(similarity_matrix, output_dir):
    """Determine number of factors using scree plot."""
    print("\n" + "=" * 60)
    print("STEP 10: Determining number of factors")
    print("=" * 60)

    if similarity_matrix is None or similarity_matrix.shape[0] < 3:
        print("Insufficient data for factor analysis")
        return 0, []

    # Initial factor analysis
    n_max = min(similarity_matrix.shape[0], similarity_matrix.shape[1])
    fa_initial = FactorAnalyzer(n_factors=n_max, rotation=None)
    fa_initial.fit(similarity_matrix)
    eigenvalues = fa_initial.get_eigenvalues()[0]

    # Kaiser criterion
    n_factors = np.count_nonzero(eigenvalues > EIGENVALUE_THRESHOLD)
    n_factors = max(2, min(n_factors, 20))  # Between 2 and 20

    print(f"Factors with eigenvalue > {EIGENVALUE_THRESHOLD}: {n_factors}")

    # Top eigenvalues
    print("\nTop 10 eigenvalues:")
    for i, ev in enumerate(eigenvalues[:10]):
        print(f"  Factor {i+1}: {ev:.2f}")

    # Scree plot
    plt.figure(figsize=(10, 6))
    x = range(1, min(30, len(eigenvalues)) + 1)
    plt.scatter(x, eigenvalues[:len(x)])
    plt.plot(x, eigenvalues[:len(x)])
    plt.axhline(y=1, color='r', linestyle='--', label='Eigenvalue = 1')
    plt.title('Scree Plot - Factor Analysis (2008-2026)')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'scree_plot_2026.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return n_factors, eigenvalues


def run_factor_analysis(similarity_matrix, n_factors):
    """Run factor analysis with Promax rotation."""
    print("\n" + "=" * 60)
    print(f"STEP 11: Factor analysis ({n_factors} factors)")
    print("=" * 60)

    if similarity_matrix is None or n_factors < 2:
        return None, None

    fa = FactorAnalyzer(n_factors=n_factors, rotation="promax")
    fa.fit(similarity_matrix)

    loadings = fa.loadings_
    print(f"Loadings matrix: {loadings.shape}")

    col_names = [f"factor{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(loadings, columns=col_names)

    return loadings_df, fa


# =============================================================================
# STEP 12: EXTRACT RESEARCH STREAMS
# =============================================================================

def extract_research_streams(loadings_df, dictionary_df, threshold, output_dir):
    """Extract papers for each research stream."""
    print("\n" + "=" * 60)
    print(f"STEP 12: Extracting research streams (threshold={threshold})")
    print("=" * 60)

    if loadings_df is None or len(dictionary_df) == 0:
        return {}, None

    results = {}

    print("\nPapers per factor:")
    print("-" * 40)

    for col in loadings_df.columns:
        mask = loadings_df[col] > threshold
        indices = loadings_df[mask].index
        n_papers = len(indices)

        valid_indices = [i for i in indices if i < len(dictionary_df)]
        if valid_indices:
            papers = dictionary_df.iloc[valid_indices]
            results[col] = papers
            papers.to_csv(os.path.join(output_dir, f"{col}.csv"),
                         index=False, encoding='utf-8')
        else:
            results[col] = pd.DataFrame()

        print(f"  {col}: {n_papers} papers")

    # Overlap matrix
    binary_df = loadings_df.applymap(lambda x: 1 if x > threshold else 0)
    overlap = binary_df.T.dot(binary_df)
    overlap.to_csv(os.path.join(output_dir, "factor_overlap.csv"))

    return results, overlap


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete analysis pipeline."""
    print("\n" + "=" * 60)
    print("OLIF CITATION ANALYSIS 2008-2026")
    print("=" * 60)
    print(f"Working directory: {WORKING_DIR}")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Step 1: Load data
    df = load_and_standardize_data()

    # Step 2: Filter
    df_filtered = filter_data(df)

    if len(df_filtered) == 0:
        print("\nERROR: No data after filtering!")
        return

    # Step 3: Extract references
    df_references = extract_references(df_filtered)

    # Step 4: Build reference database (informational)
    ref_titles = build_reference_database(df_references)

    # Step 5: Match citations within dataset
    df_pair = match_self_citations(df_references)

    if len(df_pair) == 0:
        print("\nERROR: No citation pairs found!")
        return

    # Step 6: Remove generic titles
    df_clean = remove_generic_titles(df_pair, BANNED_TITLES)

    # Step 7: Build dictionary
    dictionary_df, dictionary_full = build_title_dictionary(df_clean)
    n_citing = len(dictionary_df)

    # Step 8: Adjacency matrix
    adj_matrix, df_indexed, title_index = create_adjacency_matrix(df_clean, dictionary_full)

    # Step 9: Similarity matrix
    similarity_matrix, bc_matrix = calculate_similarity_matrix(adj_matrix, n_citing)

    # Step 10: Determine factors
    n_factors, eigenvalues = determine_n_factors(similarity_matrix, OUTPUT_DIR)

    # Step 11: Factor analysis
    loadings_df, fa = run_factor_analysis(similarity_matrix, n_factors)

    if loadings_df is not None:
        loadings_df.to_csv(os.path.join(OUTPUT_DIR, "factor_loadings.csv"))

    # Step 12: Extract streams
    results, overlap = extract_research_streams(
        loadings_df, dictionary_df, FACTOR_LOADING_THRESHOLD, OUTPUT_DIR
    )

    # Summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")

    return {
        'df': df_filtered,
        'loadings': loadings_df,
        'dictionary': dictionary_df,
        'results': results
    }


if __name__ == "__main__":
    results = main()
