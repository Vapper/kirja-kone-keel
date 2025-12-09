# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Estonian corpus linguistics tool analyzing historical newspaper texts for "kirjakeel" (written language) and "kõnekeel" (spoken language) usage patterns. Combines deterministic NLP (EstNLTK) with LLM-based semantic analysis (OpenRouter).

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# CLI usage
python main.py stats --corpus kirjakeel
python main.py collocations --corpus kirjakeel --term kirjakeel --decade 1930s
python main.py analyze --corpus kirjakeel --term kirjakeel --model anthropic/claude-sonnet-4
python main.py decades --corpus konekeel

# Launch web interface
python app.py [--port 7860] [--share]
```

## Architecture

```
src/
├── config.py        # Configuration (OpenRouter, analysis settings)
├── data_loader.py   # Excel loading, decade extraction, statistics
├── nlp_analyzer.py  # EstNLTK: morphology, POS tagging, collocations
├── llm_client.py    # OpenRouter API client (OpenAI-compatible)
├── prompts.py       # Loads and formats prompt templates
├── analyzer.py      # Combined analysis pipeline
├── interface.py     # Gradio web UI for researchers
└── output.py        # Excel/JSON export

prompts/             # LLM prompt templates (edit without touching code)
├── system.txt       # System prompt for LLM context
├── topic.txt        # Topic extraction prompt
├── tone.txt         # Tone classification prompt
├── register.txt     # Register classification prompt
├── ocr_quality.txt  # OCR quality assessment prompt
├── collocation_interpretation.txt
├── batch_topic.txt
└── summary.txt
```

## Prompt Templates

Templates in `prompts/` use `{variable}` placeholders:
- `{term}` - search term (kirjakeel/kõnekeel)
- `{decade}` - decade label (1930s, etc.)
- `{context}` / `{contexts}` - text excerpt(s)
- `{nouns}`, `{adjectives}`, `{verbs}` - collocation word lists

Edit prompts directly in `prompts/*.txt` files. Call `reload_prompts()` to refresh cached templates.

## Task Distribution

**EstNLTK (deterministic):** tokenization, morphological analysis, POS tagging, collocation extraction, NER

**LLM (semantic):** topic summarization, tone classification (negatiivne/neutraalne/positiivne), register classification (informaalne/neutraalne/formaalne), OCR quality assessment

## Data

Excel files in `andmed/` with columns: `id`, `DocumentTitle`, `date`, `section`, `context`
- `kirjakeel.xlsx` / `kirjakeel_lemmad.xlsx` (~19k rows)
- `konekeel.xlsx` / `konekeel_lemmad.xlsx` (~9k rows)

## Configuration

Set `OPENROUTER_API_KEY` in `.env` file. Models can be switched at runtime via CLI or web interface.
