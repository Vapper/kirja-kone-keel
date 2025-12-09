"""
Combined analysis pipeline using EstNLTK and LLM.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .config import TARGET_TERMS, TONE_OPTIONS, REGISTER_OPTIONS
from .data_loader import (
    load_corpus,
    group_by_decade,
    get_statistics,
    get_contexts_for_decade,
)
from .nlp_analyzer import NLPAnalyzer, CollocationResult, get_analyzer
from .llm_client import LLMClient, get_llm_client
from .prompts import (
    format_topic_prompt,
    format_tone_prompt,
    format_register_prompt,
    format_ocr_prompt,
)


@dataclass
class DecadeAnalysis:
    """Analysis results for a single decade."""
    decade: str
    article_count: int = 0
    sentence_count: int = 0
    collocations: Optional[CollocationResult] = None
    topics: list[str] = field(default_factory=list)
    tone_distribution: dict[str, int] = field(default_factory=dict)
    register_distribution: dict[str, int] = field(default_factory=dict)
    ocr_quality: dict[str, int] = field(default_factory=dict)


@dataclass
class CorpusAnalysisResult:
    """Complete analysis results for a corpus."""
    corpus_name: str
    term: str
    total_articles: int
    total_sentences: int
    decades: dict[str, DecadeAnalysis] = field(default_factory=dict)
    model_used: str = ""


class CorpusAnalyzer:
    """
    Main analyzer combining deterministic NLP and LLM-based analysis.
    """

    def __init__(
        self,
        nlp: Optional[NLPAnalyzer] = None,
        llm: Optional[LLMClient] = None,
    ):
        """
        Initialize the analyzer.

        Args:
            nlp: NLP analyzer instance (created if not provided)
            llm: LLM client instance (created if not provided)
        """
        self.nlp = nlp or get_analyzer()
        self.llm = llm  # Can be None for stats-only mode

    def compute_statistics(self, corpus_name: str) -> dict:
        """
        Compute basic statistics for a corpus (no LLM needed).

        Args:
            corpus_name: "kirjakeel" or "konekeel"

        Returns:
            Statistics dictionary
        """
        df = load_corpus(corpus_name)
        return get_statistics(df)

    def extract_collocations_by_decade(
        self,
        corpus_name: str,
        term: str,
        use_lemmatized: bool = True,
        show_progress: bool = True,
    ) -> dict[str, CollocationResult]:
        """
        Extract collocations for each decade (EstNLTK only, no LLM).

        Args:
            corpus_name: "kirjakeel" or "konekeel"
            term: Target term to find collocations for
            use_lemmatized: Whether to use lemmatized corpus
            show_progress: Show progress bar

        Returns:
            Dictionary mapping decades to CollocationResult
        """
        df = load_corpus(corpus_name, lemmatized=use_lemmatized)
        decades_data = group_by_decade(df)

        results = {}
        iterator = tqdm(decades_data.items(), desc="Extracting collocations") if show_progress else decades_data.items()

        for decade, decade_df in iterator:
            contexts = decade_df["context"].dropna().tolist()
            result = self.nlp.get_collocations_by_category(contexts, term, decade)
            results[decade] = result

        return results

    def classify_tone(self, context: str, term: str) -> str:
        """
        Classify the tone of a single context using LLM.

        Args:
            context: Text to classify
            term: The term being discussed

        Returns:
            Tone classification
        """
        if not self.llm:
            raise ValueError("LLM client not initialized")

        prompt = format_tone_prompt(term, context)
        response = self.llm.analyze(prompt)

        # Normalize response to match options
        response_lower = response.lower().strip()
        for option in TONE_OPTIONS:
            if option.lower() in response_lower or response_lower in option.lower():
                return option

        return "neutraalne"  # Default

    def classify_register(self, context: str) -> str:
        """
        Classify the register of a single context using LLM.

        Args:
            context: Text to classify

        Returns:
            Register classification
        """
        if not self.llm:
            raise ValueError("LLM client not initialized")

        prompt = format_register_prompt(context)
        response = self.llm.analyze(prompt)

        response_lower = response.lower().strip()
        for option in REGISTER_OPTIONS:
            if option.lower() in response_lower or response_lower in option.lower():
                return option

        return "neutraalne"

    def check_ocr_quality(self, context: str) -> bool:
        """
        Check if text is readable (not too corrupted by OCR).

        Args:
            context: Text to check

        Returns:
            True if readable, False if corrupted
        """
        if not self.llm:
            # Fallback: simple heuristic
            if not context or len(context) < 10:
                return False
            # Check for too many non-Estonian characters
            estonian_chars = set("abcdefghijklmnopqrsšzžtuvwõäöüxyzABCDEFGHIJKLMNOPQRSŠZŽTUVWÕÄÖÜXYZ")
            text_chars = set(c for c in context if c.isalpha())
            if len(text_chars) == 0:
                return False
            ratio = len(text_chars & estonian_chars) / len(text_chars)
            return ratio > 0.8

        prompt = format_ocr_prompt(context)
        response = self.llm.analyze(prompt)

        return "readable" in response.lower()

    def extract_topics(
        self,
        contexts: list[str],
        term: str,
        decade: str,
        max_contexts: int = 10,
    ) -> list[str]:
        """
        Extract topics from contexts using LLM.

        Args:
            contexts: List of text contexts
            term: Term being analyzed
            decade: Decade label
            max_contexts: Max contexts to include in prompt

        Returns:
            List of topic strings
        """
        if not self.llm:
            raise ValueError("LLM client not initialized")

        prompt = format_topic_prompt(term, decade, contexts, max_contexts)
        result = self.llm.analyze_json(prompt)

        if result and "topics" in result:
            return result["topics"][:5]

        return []

    def run_full_analysis(
        self,
        corpus_name: str,
        term: str,
        sample_size: int = 10,
        skip_llm: bool = False,
        show_progress: bool = True,
    ) -> CorpusAnalysisResult:
        """
        Run complete analysis on a corpus.

        Args:
            corpus_name: "kirjakeel" or "konekeel"
            term: Target term to analyze
            sample_size: Number of contexts to sample per decade for LLM analysis
            skip_llm: If True, only do deterministic analysis
            show_progress: Show progress bars

        Returns:
            Complete analysis results
        """
        # Basic statistics
        stats = self.compute_statistics(corpus_name)

        result = CorpusAnalysisResult(
            corpus_name=corpus_name,
            term=term,
            total_articles=stats["total_articles"],
            total_sentences=stats["total_sentences"],
            model_used=self.llm.model if self.llm else "none",
        )

        # Get decades
        df = load_corpus(corpus_name)
        decades_data = group_by_decade(df)

        # Process each decade
        decade_list = list(decades_data.keys())
        iterator = tqdm(decade_list, desc="Analyzing decades") if show_progress else decade_list

        for decade in iterator:
            decade_df = decades_data[decade]
            contexts = decade_df["context"].dropna().tolist()

            decade_analysis = DecadeAnalysis(
                decade=decade,
                article_count=len(decade_df),
                sentence_count=stats["by_decade"].get(decade, {}).get("sentences", 0),
            )

            # Collocation analysis (EstNLTK)
            decade_analysis.collocations = self.nlp.get_collocations_by_category(
                contexts, term, decade
            )

            # LLM-based analysis
            if not skip_llm and self.llm:
                # Sample contexts for LLM
                sample = contexts[:sample_size] if len(contexts) > sample_size else contexts

                # Topic extraction
                decade_analysis.topics = self.extract_topics(sample, term, decade)

                # Tone and register classification
                tone_counts = {opt: 0 for opt in TONE_OPTIONS}
                register_counts = {opt: 0 for opt in REGISTER_OPTIONS}
                ocr_counts = {"readable": 0, "corrupted": 0}

                for ctx in sample:
                    # OCR check first
                    if self.check_ocr_quality(ctx):
                        ocr_counts["readable"] += 1
                        tone = self.classify_tone(ctx, term)
                        register = self.classify_register(ctx)
                        tone_counts[tone] = tone_counts.get(tone, 0) + 1
                        register_counts[register] = register_counts.get(register, 0) + 1
                    else:
                        ocr_counts["corrupted"] += 1

                decade_analysis.tone_distribution = tone_counts
                decade_analysis.register_distribution = register_counts
                decade_analysis.ocr_quality = ocr_counts

            result.decades[decade] = decade_analysis

        return result

    def run_statistics_only(self, corpus_name: str) -> dict:
        """
        Run only statistical analysis (no NLP or LLM).

        Args:
            corpus_name: "kirjakeel" or "konekeel"

        Returns:
            Statistics dictionary with decade breakdown
        """
        return self.compute_statistics(corpus_name)

    def run_collocation_only(
        self,
        corpus_name: str,
        term: str,
        decade: Optional[str] = None,
    ) -> dict[str, CollocationResult] | CollocationResult:
        """
        Run only collocation analysis for specific decade or all decades.

        Args:
            corpus_name: "kirjakeel" or "konekeel"
            term: Target term
            decade: Specific decade or None for all

        Returns:
            Collocation results
        """
        if decade:
            contexts = get_contexts_for_decade(corpus_name, decade, lemmatized=True)
            return self.nlp.get_collocations_by_category(contexts, term, decade)

        return self.extract_collocations_by_decade(corpus_name, term)
