#!/usr/bin/env python3
"""
CLI entry point for corpus analysis.
"""
import argparse
import sys

from src.config import DEFAULT_MODEL, AVAILABLE_MODELS, TARGET_TERMS
from src.data_loader import get_available_decades
from src.nlp_analyzer import get_analyzer, ESTNLTK_AVAILABLE
from src.llm_client import get_llm_client
from src.analyzer import CorpusAnalyzer
from src.output import to_excel, to_json, generate_summary_report


def cmd_stats(args):
    """Show corpus statistics."""
    analyzer = CorpusAnalyzer(nlp=get_analyzer(), llm=None)
    stats = analyzer.compute_statistics(args.corpus)

    print(f"\n=== Korpus: {args.corpus} ===\n")
    print(f"Artikleid kokku: {stats['total_articles']}")
    print(f"Lauseid kokku: {stats['total_sentences']}")
    print("\nKümnendite kaupa:")
    print("-" * 40)

    for decade, data in sorted(stats["by_decade"].items()):
        print(f"  {decade}: {data['articles']:5d} artiklit, {data['sentences']:6d} lauset")

    if stats["by_section"]:
        print("\nSektsioonide kaupa:")
        print("-" * 40)
        for section, count in sorted(stats["by_section"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {section}: {count}")


def cmd_collocations(args):
    """Extract collocations."""
    if not ESTNLTK_AVAILABLE:
        print("Error: EstNLTK not installed. Run: pip install estnltk")
        sys.exit(1)

    analyzer = CorpusAnalyzer(nlp=get_analyzer(), llm=None)

    print(f"\n=== Kollokatsioonid: {args.term} ({args.corpus}) ===\n")

    if args.decade:
        result = analyzer.run_collocation_only(args.corpus, args.term, args.decade)
        print(f"Kümnend: {args.decade}\n")
        _print_collocations(result)
    else:
        results = analyzer.extract_collocations_by_decade(args.corpus, args.term)
        for decade, result in sorted(results.items()):
            print(f"\n--- {decade} ---")
            _print_collocations(result)


def _print_collocations(result):
    """Print collocation results."""
    categories = [
        ("Nimisõnad", result.nouns),
        ("Omadussõnad", result.adjectives),
        ("Tegusõnad", result.verbs),
        ("Määrsõnad", result.adverbs),
        ("Kohanimed", result.proper_nouns),
    ]

    for name, colls in categories:
        if colls:
            print(f"\n{name}:")
            for c in colls:
                print(f"  {c.word} ({c.lemma}): {c.count}x [{c.frequency}]")


def cmd_analyze(args):
    """Run full analysis with LLM."""
    print(f"\n=== Täisanalüüs: {args.term} ({args.corpus}) ===")
    print(f"Mudel: {args.model}")
    print(f"Valimi suurus: {args.sample}\n")

    nlp = get_analyzer()
    llm = get_llm_client(args.model)
    analyzer = CorpusAnalyzer(nlp=nlp, llm=llm)

    results = analyzer.run_full_analysis(
        corpus_name=args.corpus,
        term=args.term,
        sample_size=args.sample,
        skip_llm=False,
        show_progress=True,
    )

    # Print summary
    print("\n" + generate_summary_report(results))

    # Export if requested
    if args.output:
        if args.output.endswith(".json"):
            path = to_json(results, args.output)
        else:
            path = to_excel(results, args.output)
        print(f"\nTulemused salvestatud: {path}")


def cmd_decades(args):
    """List available decades in corpus."""
    decades = get_available_decades(args.corpus)
    print(f"\n=== Kümnendid korpuses: {args.corpus} ===\n")
    for decade in decades:
        print(f"  {decade}")


def main():
    parser = argparse.ArgumentParser(
        description="Kirjakeel/Kõnekeel korpuse analüüs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Näited:
  python main.py stats --corpus kirjakeel
  python main.py collocations --corpus kirjakeel --term kirjakeel --decade 1930s
  python main.py analyze --corpus kirjakeel --term kirjakeel --model anthropic/claude-sonnet-4
  python main.py decades --corpus konekeel
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Käsk")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Näita statistikat")
    stats_parser.add_argument(
        "--corpus",
        choices=["kirjakeel", "konekeel"],
        default="kirjakeel",
        help="Korpus (default: kirjakeel)",
    )

    # Collocations command
    coll_parser = subparsers.add_parser("collocations", help="Leia kollokatsioonid")
    coll_parser.add_argument(
        "--corpus",
        choices=["kirjakeel", "konekeel"],
        default="kirjakeel",
        help="Korpus",
    )
    coll_parser.add_argument(
        "--term",
        choices=TARGET_TERMS,
        default="kirjakeel",
        help="Otsitav termin",
    )
    coll_parser.add_argument(
        "--decade",
        help="Konkreetne kümnend (nt 1930s)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Täisanalüüs LLM-iga")
    analyze_parser.add_argument(
        "--corpus",
        choices=["kirjakeel", "konekeel"],
        default="kirjakeel",
        help="Korpus",
    )
    analyze_parser.add_argument(
        "--term",
        choices=TARGET_TERMS,
        default="kirjakeel",
        help="Otsitav termin",
    )
    analyze_parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default=DEFAULT_MODEL,
        help=f"LLM mudel (default: {DEFAULT_MODEL})",
    )
    analyze_parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Valimi suurus kümnendi kohta (default: 10)",
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Väljundfaili tee (.xlsx või .json)",
    )

    # Decades command
    decades_parser = subparsers.add_parser("decades", help="Näita saadaolevaid kümnendeid")
    decades_parser.add_argument(
        "--corpus",
        choices=["kirjakeel", "konekeel"],
        default="kirjakeel",
        help="Korpus",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "stats": cmd_stats,
        "collocations": cmd_collocations,
        "analyze": cmd_analyze,
        "decades": cmd_decades,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nKatkestatud.")
        sys.exit(130)
    except Exception as e:
        print(f"\nViga: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
