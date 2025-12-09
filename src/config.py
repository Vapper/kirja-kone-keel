"""
Configuration settings for the corpus analysis tool.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "andmed"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Data files
CORPUS_FILES = {
    "kirjakeel": {
        "raw": DATA_DIR / "kirjakeel.xlsx",
        "lemmatized": DATA_DIR / "kirjakeel_lemmad.xlsx",
    },
    "konekeel": {
        "raw": DATA_DIR / "konekeel.xlsx",
        "lemmatized": DATA_DIR / "konekeel_lemmad.xlsx",
    },
}

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")

# Available models for the interface
AVAILABLE_MODELS = [
    "anthropic/claude-sonnet-4",
    "anthropic/claude-haiku",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-exp",
    "google/gemini-2.5-pro-preview",
]

# Analysis settings
COLLOCATION_WINDOW = 5  # Words before/after target term
MIN_FREQUENCY = 2  # Minimum occurrence count for collocations
TOP_COLLOCATIONS = 5  # Number of top collocations per word class

# EstNLTK POS tag mappings
POS_TAGS = {
    "nouns": "S",        # Substantiiv (noun)
    "adjectives": "A",   # Adjektiiv (adjective)
    "verbs": "V",        # Verb
    "adverbs": "D",      # Adverb
    "proper_nouns": "H", # Pärisnimi (proper noun, for place names)
}

# Target terms for analysis
TARGET_TERMS = ["kirjakeel", "kõnekeel"]

# Classification categories
TONE_OPTIONS = ["negatiivne/kriitiline", "neutraalne", "positiivne"]
REGISTER_OPTIONS = ["informaalne", "neutraalne", "formaalne"]
