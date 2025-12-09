"""
Prompt templates for LLM-based analysis.
Loads templates from prompts/ directory for easy editing.
"""
from functools import lru_cache
from pathlib import Path

# Prompts directory
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=20)
def load_prompt(name: str) -> str:
    """
    Load a prompt template from file.

    Args:
        name: Template name (without .txt extension)

    Returns:
        Template string
    """
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    return path.read_text(encoding="utf-8")


def reload_prompts():
    """Clear the prompt cache to reload templates from disk."""
    load_prompt.cache_clear()


# Convenience accessors
def get_system_prompt() -> str:
    return load_prompt("system")


def get_topic_prompt() -> str:
    return load_prompt("topic")


def get_tone_prompt() -> str:
    return load_prompt("tone")


def get_register_prompt() -> str:
    return load_prompt("register")


def get_ocr_quality_prompt() -> str:
    return load_prompt("ocr_quality")


def get_collocation_interpretation_prompt() -> str:
    return load_prompt("collocation_interpretation")


def get_batch_topic_prompt() -> str:
    return load_prompt("batch_topic")


def get_summary_prompt() -> str:
    return load_prompt("summary")


# Formatting functions
def format_topic_prompt(term: str, decade: str, contexts: list[str], max_contexts: int = 10) -> str:
    """Format the topic extraction prompt with contexts."""
    selected = contexts[:max_contexts]
    formatted = "\n---\n".join([
        f"[{i+1}] {ctx[:500]}..." if len(ctx) > 500 else f"[{i+1}] {ctx}"
        for i, ctx in enumerate(selected)
    ])

    return get_topic_prompt().format(
        decade=decade,
        term=term,
        contexts=formatted
    )


def format_tone_prompt(term: str, context: str) -> str:
    """Format the tone classification prompt."""
    if len(context) > 1000:
        context = context[:1000] + "..."

    return get_tone_prompt().format(term=term, context=context)


def format_register_prompt(context: str) -> str:
    """Format the register classification prompt."""
    if len(context) > 1000:
        context = context[:1000] + "..."

    return get_register_prompt().format(context=context)


def format_ocr_prompt(context: str) -> str:
    """Format the OCR quality assessment prompt."""
    if len(context) > 500:
        context = context[:500] + "..."

    return get_ocr_quality_prompt().format(context=context)


def format_collocation_interpretation_prompt(
    term: str,
    decade: str,
    nouns: list[str],
    adjectives: list[str],
    verbs: list[str]
) -> str:
    """Format the collocation interpretation prompt."""
    return get_collocation_interpretation_prompt().format(
        term=term,
        decade=decade,
        nouns=", ".join(nouns[:5]) if nouns else "(puudub)",
        adjectives=", ".join(adjectives[:5]) if adjectives else "(puudub)",
        verbs=", ".join(verbs[:5]) if verbs else "(puudub)",
    )


def format_summary_prompt(
    term: str,
    collocations: str,
    topics: str,
    tones: str
) -> str:
    """Format the summary generation prompt."""
    return get_summary_prompt().format(
        term=term,
        collocations=collocations,
        topics=topics,
        tones=tones,
    )


def list_available_prompts() -> list[str]:
    """List all available prompt templates."""
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]
