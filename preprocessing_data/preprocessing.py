from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import csr_matrix


def prepare_data(csv_path=None):

    if csv_path is None:
        csv_path = Path(__file__).resolve().parent.parent / "data_clean.csv"

    print(f"Reading dataset from: {csv_path}")

    # Load dataset
    df = pd.read_csv(csv_path)

    # ==========================
    # TF-IDF
    # ==========================
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=False,
        strip_accents="unicode",
    )

    X = vectorizer.fit_transform(df["clean_review"].fillna(""))

    # ==========================
    # Target
    # ==========================
    y = df["label"]

    # Hapus data tanpa label (jika ada)
    mask = y.notna()
    indices = np.where(mask)[0]

    X = X[indices]
    y = y[mask].reset_index(drop=True)

    print("=" * 50)
    print(f"Total samples      : {len(df)}")
    print(f"Samples with label : {len(y)}")
    print(f"Neutral removed    : {len(df)-len(y)}")
    print("=" * 50)

    # ==========================
    # Train / Test Split
    # ==========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # ==========================
    # Train / Validation Split
    # ==========================
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )

    # ==========================
    # Random Under Sampling
    # ==========================
    rus = RandomUnderSampler(
        random_state=42,
        sampling_strategy="auto",
    )

    X_train_dense = X_train_split.toarray()

    X_train_balanced, y_train_balanced = rus.fit_resample(
        X_train_dense,
        y_train_split,
    )

    X_train_balanced = csr_matrix(X_train_balanced)

    print("\nAfter RandomUnderSampler")
    print("=" * 50)
    print(y_train_balanced.value_counts())

    print(f"\nTraining samples   : {X_train_balanced.shape[0]}")
    print(f"Validation samples : {X_val.shape[0]}")
    print(f"Test samples       : {X_test.shape[0]}")

    return (
        X_train_balanced,
        X_val,
        X_test,
        y_train_balanced,
        y_val,
        y_test,
        vectorizer,
    )