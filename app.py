#!/usr/bin/env python3
"""
Launch the Gradio researcher interface.
"""
import argparse

from src.interface import ResearcherInterface


def main():
    parser = argparse.ArgumentParser(description="Käivita korpuse analüüsi veebiliides")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host aadress (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Loo avalik jagamislink",
    )

    args = parser.parse_args()

    print("\n=== Kirjakeel/Kõnekeel Korpuse Analüüs ===\n")
    print(f"Käivitamine aadressil: http://{args.host}:{args.port}")
    if args.share:
        print("Luuakse avalik jagamislink...")
    print()

    interface = ResearcherInterface()
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
