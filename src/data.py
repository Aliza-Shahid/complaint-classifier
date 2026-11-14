"""Data preparation for the CFPB complaint classifier.

The same functions are used by the notebooks, the training scripts and predict.py,
so the cleaning rules cannot drift apart.
"""
import re
from pathlib import Path

import numpy as np  # NEW
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # NEW
from sklearn.model_selection import train_test_split

SEED = 42
CAP_PER_CLASS = 20_000
PREFIX_LEN = 150  # complaints sharing their first 150 normalised characters = template copies
NEAR_DUP_THRESHOLD = 0.80  # NEW: TF-IDF cosine similarity at/above which two complaints are near-copies
MIN_REAL_WORDS = 5  # NEW: complaints with fewer real words (placeholders excluded) carry no signal

NARRATIVE_COL = "Consumer complaint narrative"
RAW_COLUMNS = ["Complaint ID", "Date received", "Product", "Issue", NARRATIVE_COL]

# CFPB product name -> short label. The ORDER defines the label ids, so never
# reorder this after a model has been trained.
PRODUCT_TO_LABEL = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card": "Credit card",
    "Checking or savings account": "Checking or savings account",
    "Mortgage": "Mortgage",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
}
LABELS = list(PRODUCT_TO_LABEL.values())
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

_DATE_RE = re.compile(r"\bXX/(?:XX/)?(?:XXXX|\d{2,4})\b")
_REDACT_RE = re.compile(r"\bX{2,}\b")
_REDACT_RUN_RE = re.compile(r"\[REDACTED\](?:\s+\[REDACTED\])+")
_WS_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"\[(?:REDACTED|DATE)\]")  # NEW


def load_narratives(zip_path, cache_path) -> pd.DataFrame:
    """Return only complaints that have a narrative, with the text in a 'text' column.
    The filtered result is cached as a CSV (gitignored) so the zip is read once."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype=str, keep_default_na=False)
    else:
        parts = []
        for chunk in pd.read_csv(zip_path, compression="zip", usecols=RAW_COLUMNS,
                                 dtype=str, chunksize=200_000):
            chunk[NARRATIVE_COL] = chunk[NARRATIVE_COL].fillna("").str.strip()
            parts.append(chunk[chunk[NARRATIVE_COL] != ""])
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(cache_path, index=False)
    return df.rename(columns={NARRATIVE_COL: "text"})


def normalize_for_dedup(text: pd.Series) -> pd.Series:
    """Lower-case and collapse whitespace; used only to find duplicates."""
    return text.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def clean_text(text: str) -> str:
    """Model-input cleaning: make CFPB redactions consistent, collapse whitespace.
    Case is kept (the BiLSTM vectorizer and uncased DistilBERT lower-case themselves)."""
    text = _DATE_RE.sub("[DATE]", text)
    text = _REDACT_RE.sub("[REDACTED]", text)
    text = _REDACT_RUN_RE.sub("[REDACTED]", text)
    return _WS_RE.sub(" ", text).strip()


def build_dataset(raw: pd.DataFrame, seed: int = SEED, cap: int = CAP_PER_CLASS):
    """Raw narratives -> (clean dataset, report of row counts after each step).

    Steps: drop texts filed under more than one product (ambiguous label), keep one
    complaint per template opening, keep the selected products, cap every class,
    then clean the text. Survivors are chosen at random with a fixed seed."""
    df = raw.copy()
    df["norm"] = normalize_for_dedup(df["text"])
    df["prefix"] = df["norm"].str[:PREFIX_LEN]
    report = {"raw_rows": len(df)}

    n_products = df.groupby("norm")["Product"].nunique()
    conflicts = n_products[n_products > 1].index

    df = df.sample(frac=1, random_state=seed)            # random survivor choice
    df = df[~df["norm"].isin(conflicts)]
    report["after_label_conflicts"] = len(df)

    df = df.drop_duplicates(subset="prefix", keep="first")
    report["after_template_dedup"] = len(df)

    df = df[df["Product"].isin(PRODUCT_TO_LABEL)]
    report["after_product_filter"] = len(df)

    df = df.groupby("Product", group_keys=False).head(cap)

    out = pd.DataFrame({
        "complaint_id": df["Complaint ID"],
        "date_received": df["Date received"],
        "label": df["Product"].map(PRODUCT_TO_LABEL),
        "text": df["text"].map(clean_text),
    })
    out["label_id"] = out["label"].map(LABEL_TO_ID)
    out = out.reset_index(drop=True)
    report["final_rows"] = len(out)
    return out, report


def drop_near_duplicates(df: pd.DataFrame, threshold: float = NEAR_DUP_THRESHOLD,  # NEW
                         seed: int = SEED, chunk: int = 500,
                         min_df: int = 3, max_df: float = 0.5) -> pd.DataFrame:
    """Within each class, drop a complaint if its TF-IDF cosine similarity to a complaint
    already kept is >= threshold. Runs BEFORE the train/val/test split, so near-copies
    cannot end up on both sides of the split.

    The TF-IDF here is only used to find copies (unsupervised, no labels involved);
    it is not a model feature. Order is shuffled with the seed, so which copy survives
    is random but reproducible."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)

    for _, positions in df.groupby("label").indices.items():
        sub = df.iloc[positions]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=max_df,
                              max_features=100_000, sublinear_tf=True,
                              stop_words="english")
        X = vec.fit_transform(sub["text"])
        kept_local = np.ones(X.shape[0], dtype=bool)
        for s in range(0, X.shape[0], chunk):
            e = min(s + chunk, X.shape[0])
            sims = (X[s:e] @ X[:e].T).tocoo()           # rows are L2-normalised: dot = cosine
            near = (sims.data >= threshold) & (sims.col < s + sims.row)   # earlier docs only
            earlier = {}
            for r, c in zip(sims.row[near], sims.col[near]):
                earlier.setdefault(r, []).append(c)
            for r in range(e - s):
                if any(kept_local[c] for c in earlier.get(r, ())):
                    kept_local[s + r] = False
        keep[positions] = kept_local

    return df[keep].reset_index(drop=True)


def count_real_words(text: pd.Series) -> pd.Series:  # NEW
    """Words of 2+ letters, ignoring the [REDACTED] and [DATE] placeholders."""
    return text.str.replace(_PLACEHOLDER_RE, " ", regex=True).str.count(r"[A-Za-z]{2,}")


def drop_uninformative(df: pd.DataFrame, min_words: int = MIN_REAL_WORDS) -> pd.DataFrame:  # NEW
    """Drop complaints that are mostly placeholders or only a few words long."""
    return df[count_real_words(df["text"]) >= min_words].reset_index(drop=True)


def make_splits(df: pd.DataFrame, seed: int = SEED, val_size: float = 0.15,
                test_size: float = 0.15):
    """Stratified train/validation/test split (70/15/15 by default)."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_size / (1 - test_size),
        stratify=train_val["label_id"], random_state=seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))