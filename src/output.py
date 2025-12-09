"""
Output generation for analysis results.
Exports to Excel and JSON formats.
"""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

from .config import OUTPUT_DIR
from .analyzer import CorpusAnalysisResult, DecadeAnalysis
from .nlp_analyzer import CollocationResult, Collocation


def ensure_output_dir():
    """Ensure output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collocation_to_dict(coll: Collocation) -> dict:
    """Convert Collocation to dict."""
    return {
        "word": coll.word,
        "lemma": coll.lemma,
        "pos": coll.pos,
        "count": coll.count,
        "pmi": round(coll.pmi, 3),
        "frequency": coll.frequency,
    }


def collocation_result_to_dict(result: CollocationResult) -> dict:
    """Convert CollocationResult to dict."""
    return {
        "term": result.term,
        "decade": result.decade,
        "nouns": [collocation_to_dict(c) for c in result.nouns],
        "adjectives": [collocation_to_dict(c) for c in result.adjectives],
        "verbs": [collocation_to_dict(c) for c in result.verbs],
        "adverbs": [collocation_to_dict(c) for c in result.adverbs],
        "proper_nouns": [collocation_to_dict(c) for c in result.proper_nouns],
    }


def decade_analysis_to_dict(analysis: DecadeAnalysis) -> dict:
    """Convert DecadeAnalysis to dict."""
    return {
        "decade": analysis.decade,
        "article_count": analysis.article_count,
        "sentence_count": analysis.sentence_count,
        "collocations": collocation_result_to_dict(analysis.collocations) if analysis.collocations else None,
        "topics": analysis.topics,
        "tone_distribution": analysis.tone_distribution,
        "register_distribution": analysis.register_distribution,
        "ocr_quality": analysis.ocr_quality,
    }


def results_to_dict(results: CorpusAnalysisResult) -> dict:
    """Convert full results to dict."""
    return {
        "corpus_name": results.corpus_name,
        "term": results.term,
        "total_articles": results.total_articles,
        "total_sentences": results.total_sentences,
        "model_used": results.model_used,
        "generated_at": datetime.now().isoformat(),
        "decades": {
            decade: decade_analysis_to_dict(analysis)
            for decade, analysis in results.decades.items()
        },
    }


def to_json(results: CorpusAnalysisResult, output_path: Union[str, Path]) -> Path:
    """
    Export results to JSON file.

    Args:
        results: Analysis results
        output_path: Output file path

    Returns:
        Path to created file
    """
    ensure_output_dir()
    output_path = Path(output_path)

    data = results_to_dict(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def to_excel(results: CorpusAnalysisResult, output_path: Union[str, Path]) -> Path:
    """
    Export results to Excel file with multiple sheets.

    Args:
        results: Analysis results
        output_path: Output file path

    Returns:
        Path to created file
    """
    ensure_output_dir()
    output_path = Path(output_path)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:

        # Sheet 1: Overview
        overview_data = {
            "Parameeter": [
                "Korpus",
                "Termin",
                "Artikleid kokku",
                "Lauseid kokku",
                "Kasutatud mudel",
                "Genereeritud",
            ],
            "Väärtus": [
                results.corpus_name,
                results.term,
                results.total_articles,
                results.total_sentences,
                results.model_used,
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ],
        }
        pd.DataFrame(overview_data).to_excel(writer, sheet_name="Ülevaade", index=False)

        # Sheet 2: Statistics by decade
        stats_rows = []
        for decade, analysis in sorted(results.decades.items()):
            stats_rows.append({
                "Kümnend": decade,
                "Artikleid": analysis.article_count,
                "Lauseid": analysis.sentence_count,
            })
        pd.DataFrame(stats_rows).to_excel(writer, sheet_name="Statistika", index=False)

        # Sheet 3: Collocations
        coll_rows = []
        for decade, analysis in sorted(results.decades.items()):
            if analysis.collocations:
                for category, attr in [
                    ("Nimisõna", "nouns"),
                    ("Omadussõna", "adjectives"),
                    ("Tegusõna", "verbs"),
                    ("Määrsõna", "adverbs"),
                    ("Kohanimi", "proper_nouns"),
                ]:
                    for coll in getattr(analysis.collocations, attr, []):
                        coll_rows.append({
                            "Kümnend": decade,
                            "Sõnaliik": category,
                            "Sõna": coll.word,
                            "Lemma": coll.lemma,
                            "Sagedus": coll.count,
                            "PMI": round(coll.pmi, 3),
                            "Sagedusklass": coll.frequency,
                        })
        if coll_rows:
            pd.DataFrame(coll_rows).to_excel(writer, sheet_name="Kollokatsioonid", index=False)

        # Sheet 4: Topics
        topic_rows = []
        for decade, analysis in sorted(results.decades.items()):
            for i, topic in enumerate(analysis.topics, 1):
                topic_rows.append({
                    "Kümnend": decade,
                    "Teema nr": i,
                    "Teema": topic,
                })
        if topic_rows:
            pd.DataFrame(topic_rows).to_excel(writer, sheet_name="Teemad", index=False)

        # Sheet 5: Tone distribution
        tone_rows = []
        for decade, analysis in sorted(results.decades.items()):
            row = {"Kümnend": decade}
            row.update(analysis.tone_distribution)
            tone_rows.append(row)
        if tone_rows:
            pd.DataFrame(tone_rows).to_excel(writer, sheet_name="Toon", index=False)

        # Sheet 6: Register distribution
        register_rows = []
        for decade, analysis in sorted(results.decades.items()):
            row = {"Kümnend": decade}
            row.update(analysis.register_distribution)
            register_rows.append(row)
        if register_rows:
            pd.DataFrame(register_rows).to_excel(writer, sheet_name="Register", index=False)

    return output_path


def generate_summary_report(results: CorpusAnalysisResult) -> str:
    """
    Generate a human-readable summary report.

    Args:
        results: Analysis results

    Returns:
        Markdown-formatted summary
    """
    lines = [
        f"# Korpuse '{results.corpus_name}' analüüsi kokkuvõte",
        f"",
        f"**Termin:** {results.term}",
        f"**Artikleid:** {results.total_articles}",
        f"**Lauseid:** {results.total_sentences}",
        f"**Mudel:** {results.model_used}",
        f"",
        "## Kümnendite kaupa",
        "",
    ]

    for decade, analysis in sorted(results.decades.items()):
        lines.append(f"### {decade}")
        lines.append(f"- Artikleid: {analysis.article_count}")
        lines.append(f"- Lauseid: {analysis.sentence_count}")

        if analysis.topics:
            lines.append(f"- Teemad: {', '.join(analysis.topics)}")

        if analysis.collocations:
            nouns = [c.word for c in analysis.collocations.nouns[:3]]
            if nouns:
                lines.append(f"- Sagedased nimisõnad: {', '.join(nouns)}")

        if analysis.tone_distribution:
            dominant = max(analysis.tone_distribution.items(), key=lambda x: x[1])
            lines.append(f"- Domineeriv toon: {dominant[0]}")

        lines.append("")

    return "\n".join(lines)


def save_summary_report(results: CorpusAnalysisResult, output_path: Union[str, Path]) -> Path:
    """
    Save summary report to markdown file.

    Args:
        results: Analysis results
        output_path: Output file path

    Returns:
        Path to created file
    """
    ensure_output_dir()
    output_path = Path(output_path)

    report = generate_summary_report(results)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path
