"""
Data loading utilities for corpus Excel files.
"""
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CORPUS_FILES, DATA_DIR


def load_corpus(corpus_name: str, lemmatized: bool = False) -> pd.DataFrame:
    """
    Load a corpus from Excel file.

    Args:
        corpus_name: Either "kirjakeel" or "konekeel"
        lemmatized: If True, load the lemmatized version

    Returns:
        DataFrame with columns: id, DocumentTitle, date, section, context
    """
    if corpus_name not in CORPUS_FILES:
        raise ValueError(f"Unknown corpus: {corpus_name}. Use 'kirjakeel' or 'konekeel'")

    file_key = "lemmatized" if lemmatized else "raw"
    file_path = CORPUS_FILES[corpus_name][file_key]

    df = pd.read_excel(file_path)
    return df


def extract_decade(date_str: str) -> Optional[str]:
    """
    Extract decade from date string.

    Args:
        date_str: Date in format like "19310929" or "2021-04-27"

    Returns:
        Decade string like "1930s" or None if parsing fails
    """
    if pd.isna(date_str):
        return None

    date_str = str(date_str)

    # Try to extract 4-digit year
    match = re.search(r"(1[89]\d{2}|20\d{2})", date_str)
    if match:
        year = int(match.group(1))
        decade = (year // 10) * 10
        return f"{decade}s"

    return None


def group_by_decade(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Group corpus data by decade.

    Args:
        df: DataFrame with 'date' column

    Returns:
        Dictionary mapping decade strings to DataFrames
    """
    df = df.copy()
    df["decade"] = df["date"].apply(extract_decade)

    # Filter out rows without valid decade
    df = df[df["decade"].notna()]

    grouped = {decade: group.drop(columns=["decade"])
               for decade, group in df.groupby("decade")}

    # Sort by decade
    return dict(sorted(grouped.items()))


def get_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for a corpus.

    Args:
        df: DataFrame with corpus data

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_articles": len(df),
        "total_sentences": 0,
        "by_decade": {},
        "by_section": {},
    }

    # Count sentences (rough estimate: split by period)
    if "context" in df.columns:
        stats["total_sentences"] = df["context"].apply(
            lambda x: len(str(x).split(".")) if pd.notna(x) else 0
        ).sum()

    # Group by decade
    df_with_decade = df.copy()
    df_with_decade["decade"] = df_with_decade["date"].apply(extract_decade)

    for decade, group in df_with_decade.groupby("decade"):
        if pd.notna(decade):
            sentence_count = group["context"].apply(
                lambda x: len(str(x).split(".")) if pd.notna(x) else 0
            ).sum()
            stats["by_decade"][decade] = {
                "articles": len(group),
                "sentences": sentence_count,
            }

    # Group by section
    if "section" in df.columns:
        for section, group in df.groupby("section"):
            if pd.notna(section):
                stats["by_section"][section] = len(group)

    return stats


def get_available_decades(corpus_name: str) -> list[str]:
    """
    Get list of available decades in a corpus.

    Args:
        corpus_name: Either "kirjakeel" or "konekeel"

    Returns:
        Sorted list of decade strings
    """
    df = load_corpus(corpus_name)
    decades = df["date"].apply(extract_decade).dropna().unique()
    return sorted(decades)


def get_contexts_for_decade(
    corpus_name: str,
    decade: str,
    lemmatized: bool = False,
    limit: Optional[int] = None
) -> list[str]:
    """
    Get all contexts for a specific decade.

    Args:
        corpus_name: Either "kirjakeel" or "konekeel"
        decade: Decade string like "1930s"
        lemmatized: If True, use lemmatized version
        limit: Optional limit on number of contexts

    Returns:
        List of context strings
    """
    df = load_corpus(corpus_name, lemmatized=lemmatized)
    df["decade"] = df["date"].apply(extract_decade)

    contexts = df[df["decade"] == decade]["context"].dropna().tolist()

    if limit:
        contexts = contexts[:limit]

    return contexts
