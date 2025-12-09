"""
EstNLTK-based NLP analysis for Estonian corpus data.
"""
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

from .config import COLLOCATION_WINDOW, MIN_FREQUENCY, POS_TAGS, TOP_COLLOCATIONS

# EstNLTK imports - wrapped for graceful degradation
try:
    from estnltk import Text
    from estnltk.taggers import NerTagger
    ESTNLTK_AVAILABLE = True
except ImportError:
    ESTNLTK_AVAILABLE = False
    Text = None


@dataclass
class Collocation:
    """Represents a collocating word."""
    word: str
    lemma: str
    pos: str
    count: int
    pmi: float  # Pointwise Mutual Information
    frequency: str  # "sage", "keskmine", "harva"


@dataclass
class CollocationResult:
    """Results of collocation analysis."""
    term: str
    decade: str
    nouns: list[Collocation]
    adjectives: list[Collocation]
    verbs: list[Collocation]
    adverbs: list[Collocation]
    proper_nouns: list[Collocation]  # Place names etc.


class NLPAnalyzer:
    """
    Estonian NLP analyzer using EstNLTK.
    Provides morphological analysis, collocation extraction, and NER.
    """

    def __init__(self):
        if not ESTNLTK_AVAILABLE:
            raise ImportError(
                "EstNLTK is not installed. Install with: pip install estnltk"
            )
        self._ner_tagger = None

    @property
    def ner_tagger(self):
        """Lazy-load NER tagger."""
        if self._ner_tagger is None:
            self._ner_tagger = NerTagger()
        return self._ner_tagger

    def analyze_text(self, text: str) -> "Text":
        """
        Perform full morphological analysis on text.

        Args:
            text: Estonian text to analyze

        Returns:
            EstNLTK Text object with annotations
        """
        doc = Text(text)
        doc.tag_layer(["morph_analysis"])
        return doc

    def count_sentences(self, text: str) -> int:
        """
        Count sentences in text using EstNLTK tokenization.

        Args:
            text: Text to analyze

        Returns:
            Number of sentences
        """
        doc = Text(text)
        doc.tag_layer(["sentences"])
        return len(doc.sentences)

    def extract_collocations(
        self,
        texts: list[str],
        target: str,
        window: int = COLLOCATION_WINDOW
    ) -> dict[str, dict]:
        """
        Extract words co-occurring with target term.

        Args:
            texts: List of context texts
            target: Target word (e.g., "kirjakeel")
            window: Number of words before/after to consider

        Returns:
            Dictionary: {lemma: {pos, count, words: [surface forms]}}
        """
        collocations = defaultdict(lambda: {"pos": None, "count": 0, "words": []})
        total_words = 0
        target_count = 0

        target_lower = target.lower()

        for text in texts:
            if not text or not isinstance(text, str):
                continue

            doc = Text(text)
            doc.tag_layer(["morph_analysis"])

            words = list(doc.morph_analysis)
            total_words += len(words)

            for i, word_analysis in enumerate(words):
                # Check if this word matches target (by lemma or surface form)
                word_text = word_analysis.text.lower()
                word_lemmas = [l.lower() for l in word_analysis.lemma]

                if target_lower in word_lemmas or target_lower == word_text:
                    target_count += 1

                    # Get words in window
                    start = max(0, i - window)
                    end = min(len(words), i + window + 1)

                    for j in range(start, end):
                        if j == i:
                            continue

                        neighbor = words[j]
                        neighbor_lemmas = neighbor.lemma
                        neighbor_pos_tags = neighbor.partofspeech

                        if neighbor_lemmas and neighbor_pos_tags:
                            # Use first analysis (most likely)
                            lemma = neighbor_lemmas[0]
                            pos = neighbor_pos_tags[0]

                            # Skip punctuation and short words
                            if len(lemma) < 2 or pos in ["Z", "J"]:
                                continue

                            collocations[lemma]["pos"] = pos
                            collocations[lemma]["count"] += 1
                            if neighbor.text not in collocations[lemma]["words"]:
                                collocations[lemma]["words"].append(neighbor.text)

        # Calculate PMI for each collocation
        for lemma, data in collocations.items():
            if total_words > 0 and target_count > 0:
                # PMI = log2(P(x,y) / (P(x) * P(y)))
                p_xy = data["count"] / total_words
                p_x = target_count / total_words
                p_y = data["count"] / total_words  # Approximation
                if p_x > 0 and p_y > 0:
                    data["pmi"] = math.log2(p_xy / (p_x * p_y)) if p_xy > 0 else 0
                else:
                    data["pmi"] = 0
            else:
                data["pmi"] = 0

        return dict(collocations)

    def filter_by_pos(
        self,
        collocations: dict[str, dict],
        pos_tag: str,
        top_n: int = TOP_COLLOCATIONS,
        min_freq: int = MIN_FREQUENCY
    ) -> list[Collocation]:
        """
        Filter collocations by POS tag and return top results.

        Args:
            collocations: Output from extract_collocations
            pos_tag: POS tag to filter (S, A, V, D, H)
            top_n: Number of top results
            min_freq: Minimum frequency threshold

        Returns:
            List of Collocation objects
        """
        filtered = [
            (lemma, data) for lemma, data in collocations.items()
            if data["pos"] == pos_tag and data["count"] >= min_freq
        ]

        # Sort by count (could also sort by PMI)
        filtered.sort(key=lambda x: x[1]["count"], reverse=True)

        results = []
        max_count = filtered[0][1]["count"] if filtered else 1

        for lemma, data in filtered[:top_n]:
            # Determine frequency category
            ratio = data["count"] / max_count
            if ratio > 0.6:
                freq = "sage"
            elif ratio > 0.3:
                freq = "keskmine"
            else:
                freq = "harva"

            results.append(Collocation(
                word=data["words"][0] if data["words"] else lemma,
                lemma=lemma,
                pos=data["pos"],
                count=data["count"],
                pmi=data["pmi"],
                frequency=freq
            ))

        return results

    def get_collocations_by_category(
        self,
        texts: list[str],
        target: str,
        decade: str
    ) -> CollocationResult:
        """
        Extract and categorize collocations for a target term.

        Args:
            texts: List of context texts
            target: Target word
            decade: Decade label for results

        Returns:
            CollocationResult with categorized collocations
        """
        collocations = self.extract_collocations(texts, target)

        return CollocationResult(
            term=target,
            decade=decade,
            nouns=self.filter_by_pos(collocations, POS_TAGS["nouns"]),
            adjectives=self.filter_by_pos(collocations, POS_TAGS["adjectives"]),
            verbs=self.filter_by_pos(collocations, POS_TAGS["verbs"]),
            adverbs=self.filter_by_pos(collocations, POS_TAGS["adverbs"]),
            proper_nouns=self.filter_by_pos(collocations, POS_TAGS["proper_nouns"]),
        )

    def detect_named_entities(self, text: str) -> list[dict]:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of dicts with 'text', 'label' (PER, LOC, ORG), 'start', 'end'
        """
        doc = Text(text)
        doc.tag_layer(["morph_analysis", "ner"])

        entities = []
        for entity in doc.ner:
            entities.append({
                "text": entity.text,
                "label": entity.nertag,
                "start": entity.start,
                "end": entity.end,
            })

        return entities


# Fallback analyzer when EstNLTK is not available
class SimpleNLPAnalyzer:
    """
    Simple fallback analyzer using basic string operations.
    Used when EstNLTK is not available.
    """

    def count_sentences(self, text: str) -> int:
        """Count sentences by splitting on sentence-ending punctuation."""
        if not text:
            return 0
        # Simple sentence count by periods, exclamation, question marks
        import re
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def extract_collocations(
        self,
        texts: list[str],
        target: str,
        window: int = 5
    ) -> dict[str, dict]:
        """
        Simple word co-occurrence without morphological analysis.
        """
        import re
        collocations = defaultdict(lambda: {"pos": "?", "count": 0, "words": [], "pmi": 0})
        target_lower = target.lower()

        for text in texts:
            if not text:
                continue

            words = re.findall(r'\b\w+\b', text.lower())

            for i, word in enumerate(words):
                if word == target_lower:
                    start = max(0, i - window)
                    end = min(len(words), i + window + 1)

                    for j in range(start, end):
                        if j != i and len(words[j]) > 2:
                            neighbor = words[j]
                            collocations[neighbor]["count"] += 1
                            if neighbor not in collocations[neighbor]["words"]:
                                collocations[neighbor]["words"].append(neighbor)

        return dict(collocations)


def get_analyzer() -> NLPAnalyzer | SimpleNLPAnalyzer:
    """
    Get appropriate NLP analyzer based on available dependencies.
    """
    if ESTNLTK_AVAILABLE:
        return NLPAnalyzer()
    else:
        return SimpleNLPAnalyzer()
