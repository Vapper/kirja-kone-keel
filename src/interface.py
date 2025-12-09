"""
Gradio-based researcher interface for corpus analysis.
"""
from typing import Optional
import pandas as pd

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

from .config import AVAILABLE_MODELS, DEFAULT_MODEL, TARGET_TERMS
from .data_loader import load_corpus, get_available_decades, get_statistics
from .nlp_analyzer import get_analyzer, ESTNLTK_AVAILABLE
from .llm_client import get_llm_client
from .analyzer import CorpusAnalyzer


class ResearcherInterface:
    """
    Web-based interface for language researchers.
    Provides no-code access to corpus analysis tools.
    """

    def __init__(self):
        if not GRADIO_AVAILABLE:
            raise ImportError("Gradio not installed. Run: pip install gradio")

        self.analyzer: Optional[CorpusAnalyzer] = None
        self.current_results = None

    def _init_analyzer(self, model: str, use_llm: bool = True):
        """Initialize analyzer with selected model."""
        nlp = get_analyzer()
        llm = get_llm_client(model) if use_llm else None
        self.analyzer = CorpusAnalyzer(nlp=nlp, llm=llm)

    def _get_statistics(self, corpus: str) -> pd.DataFrame:
        """Get corpus statistics as DataFrame."""
        stats = get_statistics(load_corpus(corpus))

        rows = []
        for decade, data in sorted(stats["by_decade"].items()):
            rows.append({
                "Kümnend": decade,
                "Artikleid": data["articles"],
                "Lauseid": data["sentences"],
            })

        df = pd.DataFrame(rows)

        # Add totals row
        totals = pd.DataFrame([{
            "Kümnend": "KOKKU",
            "Artikleid": stats["total_articles"],
            "Lauseid": stats["total_sentences"],
        }])
        df = pd.concat([df, totals], ignore_index=True)

        return df

    def _get_collocations(
        self,
        corpus: str,
        term: str,
        decade: str,
        word_classes: list[str],
    ) -> pd.DataFrame:
        """Get collocations as DataFrame."""
        if not self.analyzer:
            self._init_analyzer(DEFAULT_MODEL, use_llm=False)

        result = self.analyzer.run_collocation_only(corpus, term, decade)

        rows = []
        class_mapping = {
            "Nimisõnad": "nouns",
            "Omadussõnad": "adjectives",
            "Tegusõnad": "verbs",
            "Määrsõnad": "adverbs",
            "Kohanimed": "proper_nouns",
        }

        for class_name in word_classes:
            attr_name = class_mapping.get(class_name)
            if attr_name:
                collocations = getattr(result, attr_name, [])
                for coll in collocations:
                    rows.append({
                        "Sõnaliik": class_name,
                        "Sõna": coll.word,
                        "Lemma": coll.lemma,
                        "Sagedus": coll.count,
                        "Sagedusklass": coll.frequency,
                    })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["Sõnaliik", "Sõna", "Lemma", "Sagedus", "Sagedusklass"]
        )

    def _run_semantic_analysis(
        self,
        corpus: str,
        term: str,
        model: str,
        sample_size: int,
        progress=gr.Progress(),
    ) -> tuple:
        """Run LLM-based semantic analysis."""
        self._init_analyzer(model, use_llm=True)

        progress(0, desc="Alustamine...")

        result = self.analyzer.run_full_analysis(
            corpus_name=corpus,
            term=term,
            sample_size=sample_size,
            skip_llm=False,
            show_progress=False,
        )

        self.current_results = result

        # Format topics
        topics_data = {}
        for decade, analysis in sorted(result.decades.items()):
            topics_data[decade] = analysis.topics

        # Format tone distribution for plotting
        tone_rows = []
        for decade, analysis in sorted(result.decades.items()):
            for tone, count in analysis.tone_distribution.items():
                tone_rows.append({
                    "Kümnend": decade,
                    "Toon": tone,
                    "Arv": count,
                })
        tone_df = pd.DataFrame(tone_rows)

        # Format register distribution
        register_rows = []
        for decade, analysis in sorted(result.decades.items()):
            for register, count in analysis.register_distribution.items():
                register_rows.append({
                    "Kümnend": decade,
                    "Register": register,
                    "Arv": count,
                })
        register_df = pd.DataFrame(register_rows)

        return topics_data, tone_df, register_df

    def create_interface(self) -> "gr.Blocks":
        """Create the Gradio interface."""

        with gr.Blocks(
            title="Kirjakeel/Kõnekeel Korpuse Analüüs",
            theme=gr.themes.Soft(),
        ) as app:

            gr.Markdown("# Kirjakeel/Kõnekeel Korpuse Analüüs")
            gr.Markdown("Eesti ajalooliste ajalehetekstide analüüsitööriist")

            # Settings section
            with gr.Row():
                corpus_dropdown = gr.Dropdown(
                    choices=["kirjakeel", "kõnekeel"],
                    value="kirjakeel",
                    label="Korpus",
                )
                term_dropdown = gr.Dropdown(
                    choices=TARGET_TERMS,
                    value="kirjakeel",
                    label="Otsitav termin",
                )
                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    label="LLM mudel",
                )

            with gr.Tabs():

                # Tab 1: Statistics
                with gr.Tab("Statistika"):
                    gr.Markdown("### Korpuse statistika (ilma LLM-ta)")
                    stats_btn = gr.Button("Arvuta statistika", variant="primary")
                    stats_output = gr.DataFrame(label="Statistika kümnendite kaupa")

                    stats_btn.click(
                        fn=self._get_statistics,
                        inputs=[corpus_dropdown],
                        outputs=[stats_output],
                    )

                # Tab 2: Collocations
                with gr.Tab("Kollokatsioonid"):
                    gr.Markdown("### Kollokatsioonanalüüs (EstNLTK)")

                    if not ESTNLTK_AVAILABLE:
                        gr.Markdown("⚠️ EstNLTK pole installitud. Installige: `pip install estnltk`")

                    with gr.Row():
                        decade_dropdown = gr.Dropdown(
                            choices=[],
                            label="Kümnend",
                        )
                        pos_checkboxes = gr.CheckboxGroup(
                            choices=["Nimisõnad", "Omadussõnad", "Tegusõnad", "Määrsõnad", "Kohanimed"],
                            value=["Nimisõnad", "Omadussõnad"],
                            label="Sõnaliigid",
                        )

                    collocation_btn = gr.Button("Leia kollokatsioonid", variant="primary")
                    collocation_output = gr.DataFrame(label="Kollokatsioonid")

                    # Update decades when corpus changes
                    def update_decades(corpus):
                        try:
                            decades = get_available_decades(corpus)
                            return gr.Dropdown(choices=decades, value=decades[0] if decades else None)
                        except Exception:
                            return gr.Dropdown(choices=[], value=None)

                    corpus_dropdown.change(
                        fn=update_decades,
                        inputs=[corpus_dropdown],
                        outputs=[decade_dropdown],
                    )

                    collocation_btn.click(
                        fn=self._get_collocations,
                        inputs=[corpus_dropdown, term_dropdown, decade_dropdown, pos_checkboxes],
                        outputs=[collocation_output],
                    )

                # Tab 3: Semantic Analysis
                with gr.Tab("Semantiline analüüs"):
                    gr.Markdown("### LLM-põhine analüüs (OpenRouter)")
                    gr.Markdown("Teemad, toon ja register")

                    sample_slider = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Valimi suurus kümnendi kohta",
                    )

                    analyze_btn = gr.Button("Käivita analüüs", variant="primary")

                    with gr.Row():
                        topics_output = gr.JSON(label="Teemad kümnendite kaupa")

                    with gr.Row():
                        tone_output = gr.DataFrame(label="Tooni jaotus")
                        register_output = gr.DataFrame(label="Registri jaotus")

                    analyze_btn.click(
                        fn=self._run_semantic_analysis,
                        inputs=[corpus_dropdown, term_dropdown, model_dropdown, sample_slider],
                        outputs=[topics_output, tone_output, register_output],
                    )

                # Tab 4: Export
                with gr.Tab("Eksport"):
                    gr.Markdown("### Tulemuste eksport")

                    with gr.Row():
                        export_excel_btn = gr.Button("Ekspordi Excelisse")
                        export_json_btn = gr.Button("Ekspordi JSON-i")

                    export_file = gr.File(label="Allalaadimiseks")

                    def export_excel():
                        if self.current_results:
                            from .output import to_excel
                            path = to_excel(self.current_results, "output/results.xlsx")
                            return path
                        return None

                    def export_json():
                        if self.current_results:
                            from .output import to_json
                            path = to_json(self.current_results, "output/results.json")
                            return path
                        return None

                    export_excel_btn.click(fn=export_excel, outputs=[export_file])
                    export_json_btn.click(fn=export_json, outputs=[export_file])

            # Footer
            gr.Markdown("---")
            gr.Markdown("*Kirjakeel/Kõnekeel analüüsitööriist | EstNLTK + OpenRouter*")

        return app

    def launch(self, **kwargs):
        """Launch the interface."""
        app = self.create_interface()
        app.launch(**kwargs)


def create_app() -> "gr.Blocks":
    """Create and return the Gradio app."""
    interface = ResearcherInterface()
    return interface.create_interface()
