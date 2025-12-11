"""
Combined analysis pipeline using EstNLTK and LLM.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    def _is_readable_heuristic(self, context: str) -> bool:
        """Simple heuristic to check if text is readable (no LLM)."""
        if not context or len(context) < 10:
            return False
        # Check for too many non-Estonian characters
        estonian_chars = set("abcdefghijklmnopqrsšzžtuvwõäöüxyzABCDEFGHIJKLMNOPQRSŠZŽTUVWÕÄÖÜXYZ")
        text_chars = set(c for c in context if c.isalpha())
        if len(text_chars) == 0:
            return False
        ratio = len(text_chars & estonian_chars) / len(text_chars)
        return ratio > 0.8

    def _batch_classify_tone(self, contexts: list[str], term: str) -> dict[str, int]:
        """Classify tone for multiple contexts in one LLM call."""
        logger.info(f"  → Batch tone classification for {len(contexts)} contexts")
        if not contexts:
            return {opt: 0 for opt in TONE_OPTIONS}

        # Build batch prompt
        contexts_text = "\n---\n".join([f"[{i+1}] {ctx[:300]}" for i, ctx in enumerate(contexts[:10])])
        prompt = f"""Classify the tone of each Estonian text excerpt about "{term}".

{contexts_text}

For each excerpt [1], [2], etc., classify as: negatiivne/kriitiline, neutraalne, or positiivne

Return as JSON: {{"1": "tone", "2": "tone", ...}}"""

        logger.debug(f"  Sending tone prompt ({len(prompt)} chars)")
        result = self.llm.analyze_json(prompt)
        logger.debug(f"  Tone result: {result}")

        # Count tones
        tone_counts = {opt: 0 for opt in TONE_OPTIONS}
        if result:
            for key, tone in result.items():
                tone_lower = tone.lower() if isinstance(tone, str) else ""
                for option in TONE_OPTIONS:
                    if option.lower() in tone_lower or tone_lower in option.lower():
                        tone_counts[option] += 1
                        break
                else:
                    tone_counts["neutraalne"] += 1

        return tone_counts

    def _batch_classify_register(self, contexts: list[str]) -> dict[str, int]:
        """Classify register for multiple contexts in one LLM call."""
        logger.info(f"  → Batch register classification for {len(contexts)} contexts")
        if not contexts:
            return {opt: 0 for opt in REGISTER_OPTIONS}

        # Build batch prompt
        contexts_text = "\n---\n".join([f"[{i+1}] {ctx[:300]}" for i, ctx in enumerate(contexts[:10])])
        prompt = f"""Classify the register/style of each Estonian text excerpt.

{contexts_text}

For each excerpt [1], [2], etc., classify as: informaalne, neutraalne, or formaalne

Return as JSON: {{"1": "register", "2": "register", ...}}"""

        logger.debug(f"  Sending register prompt ({len(prompt)} chars)")
        result = self.llm.analyze_json(prompt)
        logger.debug(f"  Register result: {result}")

        # Count registers
        register_counts = {opt: 0 for opt in REGISTER_OPTIONS}
        if result:
            for key, register in result.items():
                register_lower = register.lower() if isinstance(register, str) else ""
                for option in REGISTER_OPTIONS:
                    if option.lower() in register_lower or register_lower in option.lower():
                        register_counts[option] += 1
                        break
                else:
                    register_counts["neutraalne"] += 1

        return register_counts

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
        logger.info(f"  → Extracting topics from {len(contexts)} contexts")
        if not self.llm:
            raise ValueError("LLM client not initialized")

        prompt = format_topic_prompt(term, decade, contexts, max_contexts)
        logger.debug(f"  Sending topic prompt ({len(prompt)} chars)")
        result = self.llm.analyze_json(prompt)
        logger.debug(f"  Topic result: {result}")

        if result and "topics" in result:
            topics = result["topics"][:5]
            logger.info(f"  ✓ Found {len(topics)} topics")
            return topics

        logger.warning(f"  ✗ No topics extracted")
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
        logger.info(f"Starting full analysis: corpus={corpus_name}, term={term}, sample_size={sample_size}")

        # Basic statistics
        logger.info("Computing statistics...")
        stats = self.compute_statistics(corpus_name)
        logger.info(f"Total articles: {stats['total_articles']}, sentences: {stats['total_sentences']}")

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
        logger.info(f"Found {len(decades_data)} decades: {list(decades_data.keys())}")

        # Process each decade
        decade_list = list(decades_data.keys())
        iterator = tqdm(decade_list, desc="Analyzing decades") if show_progress else decade_list

        for decade in iterator:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing decade: {decade}")
            decade_df = decades_data[decade]
            contexts = decade_df["context"].dropna().tolist()
            logger.info(f"  Contexts available: {len(contexts)}")

            decade_analysis = DecadeAnalysis(
                decade=decade,
                article_count=len(decade_df),
                sentence_count=stats["by_decade"].get(decade, {}).get("sentences", 0),
            )

            # Collocation analysis (EstNLTK)
            logger.info(f"  → Running collocation analysis (EstNLTK)...")
            decade_analysis.collocations = self.nlp.get_collocations_by_category(
                contexts, term, decade
            )
            logger.info(f"  ✓ Collocations extracted")

            # LLM-based analysis
            if not skip_llm and self.llm:
                logger.info(f"  → Starting LLM analysis (model: {self.llm.model})")
                # Sample contexts for LLM
                sample = contexts[:sample_size] if len(contexts) > sample_size else contexts
                logger.info(f"  Sample size: {len(sample)}")

                # Filter readable contexts using simple heuristic (no LLM)
                readable_sample = [ctx for ctx in sample if self._is_readable_heuristic(ctx)]
                ocr_counts = {
                    "readable": len(readable_sample),
                    "corrupted": len(sample) - len(readable_sample)
                }
                logger.info(f"  Readable contexts: {len(readable_sample)}/{len(sample)}")

                # Topic extraction (1 LLM call per decade)
                decade_analysis.topics = self.extract_topics(readable_sample, term, decade)

                # Batch tone and register classification (1 call each per decade)
                tone_counts = self._batch_classify_tone(readable_sample, term)
                register_counts = self._batch_classify_register(readable_sample)

                decade_analysis.tone_distribution = tone_counts
                decade_analysis.register_distribution = register_counts
                decade_analysis.ocr_quality = ocr_counts

                logger.info(f"  ✓ LLM analysis complete for {decade}")
                logger.info(f"    Topics: {decade_analysis.topics}")
                logger.info(f"    Tone: {tone_counts}")
                logger.info(f"    Register: {register_counts}")

            result.decades[decade] = decade_analysis

        logger.info(f"\n{'='*50}")
        logger.info(f"Analysis complete! Processed {len(result.decades)} decades")
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
