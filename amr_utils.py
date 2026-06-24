"""AMR parsing helpers using amrlib."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

import pandas as pd

from alignment_scorers import PairMode


class AMRParserError(Exception):
    """User-facing AMR parser failure."""


@lru_cache(maxsize=1)
def load_stog_parser():
    try:
        import amrlib

        return amrlib.load_stog_model()
    except ModuleNotFoundError as exc:
        raise AMRParserError(
            f"Missing dependency for amrlib: {exc.name}. "
            f"Run: pip install unidecode penman spacy"
        ) from exc
    except Exception as exc:
        msg = str(exc).lower()
        if "model_stog" in msg or "no such file" in msg or "not found" in msg:
            hint = (
                "Download a sentence-to-graph model from "
                "https://github.com/bjascob/amrlib-models/releases, extract it under "
                "`amrlib/data/`, and link or rename the folder to `model_stog`. "
                "See https://amrlib.readthedocs.io/en/latest/install/"
            )
        else:
            hint = f"Underlying error: {exc}"
        raise AMRParserError(hint) from exc


def parse_english_sentences(
    sentences: List[str],
    add_metadata: bool = True,
) -> List[str]:
    nonempty = [s.strip() for s in sentences if s and s.strip()]
    if not nonempty:
        return []

    parser = load_stog_parser()
    graphs = parser.parse_sents(
        nonempty,
        add_metadata=add_metadata,
        disable_progress=True,
    )
    return [str(graph) if graph else "" for graph in graphs]


def build_amr_dataframe(
    texts_1: List[str],
    texts_2: List[str],
    pair_mode: PairMode,
) -> pd.DataFrame:
    """Parse AMR graphs for all English sentences in the loaded pairs."""
    unique_sentences: List[str] = []
    seen: set[str] = set()

    if pair_mode == "de_en":
        for text in texts_2:
            sent = str(text).strip() if text is not None else ""
            if sent and sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)
    else:
        for text in list(texts_1) + list(texts_2):
            sent = str(text).strip() if text is not None else ""
            if sent and sent not in seen:
                seen.add(sent)
                unique_sentences.append(sent)

    if not unique_sentences:
        raise AMRParserError("No non-empty English sentences to parse.")

    graphs = parse_english_sentences(unique_sentences)
    graph_by_sentence = dict(zip(unique_sentences, graphs))

    rows = []
    for i, (left, right) in enumerate(zip(texts_1, texts_2)):
        left = "" if left is None else str(left)
        right = "" if right is None else str(right)

        if pair_mode == "de_en":
            en = right.strip()
            rows.append(
                {
                    "Pair": i + 1,
                    "German": left,
                    "English": right,
                    "AMR (English)": graph_by_sentence.get(en, "") if en else "",
                }
            )
        else:
            a = left.strip()
            b = right.strip()
            rows.append(
                {
                    "Pair": i + 1,
                    "Sentence A": left,
                    "Sentence B": right,
                    "AMR (A)": graph_by_sentence.get(a, "") if a else "",
                    "AMR (B)": graph_by_sentence.get(b, "") if b else "",
                }
            )

    return pd.DataFrame(rows)


def render_amr_graph_png(graph_text: str) -> Optional[bytes]:
    if not graph_text or not str(graph_text).strip():
        return None

    try:
        from amrlib.graph_processing.amr_plot import AMRPlot

        plot = AMRPlot(format="png")
        plot.build_from_graph(str(graph_text))
        return plot.graph.pipe()
    except Exception:
        return None
