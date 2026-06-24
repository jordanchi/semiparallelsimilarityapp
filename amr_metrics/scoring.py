"""AMR graph similarity metrics (SMATCH and S²MATCH)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Optional

import pandas as pd
import smatch

from alignment_scorers import PairMode


class AMRMetricError(Exception):
    """User-facing AMR metric failure."""


def normalize_amr_line(graph_text: str) -> str:
    """Convert a multi-line AMR graph string to a single smatch-parseable line."""
    lines = []
    for line in str(graph_text).splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return re.sub(r" +", " ", " ".join(lines))


def score_smatch_pair(graph_a: str, graph_b: str) -> Dict[str, float]:
    """Return SMATCH precision, recall, and F1 for two AMR graphs."""
    amr_a = normalize_amr_line(graph_a)
    amr_b = normalize_amr_line(graph_b)
    if not amr_a or not amr_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    smatch.match_triple_dict.clear()
    try:
        match_num, test_num, gold_num = smatch.get_amr_match(amr_a, amr_b)
    except Exception as exc:
        raise AMRMetricError(f"SMATCH failed to parse AMR graphs: {exc}") from exc

    precision, recall, f1 = smatch.compute_f(match_num, test_num, gold_num)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def default_glove_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.environ.get("GLOVE_VECTORS_PATH", ""),
        os.path.join(project_root, "vectors", "glove.6B.100d.txt"),
        os.path.expanduser("~/vectors/glove.6B.100d.txt"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return candidates[1]


@lru_cache(maxsize=1)
def _load_glove_vectors(path: str) -> Dict[str, object]:
    from amr_metrics.s2match_core import load_vecs

    if not path or not os.path.isfile(path):
        raise AMRMetricError(
            "S²MATCH requires GloVe vectors. Download glove.6B.100d.txt from "
            "https://nlp.stanford.edu/projects/glove/ and place it at "
            f"`{path}`, or set GLOVE_VECTORS_PATH."
        )
    return load_vecs(path)


def score_s2match_pair(
    graph_a: str,
    graph_b: str,
    vectors_path: Optional[str] = None,
    cutoff: float = 0.5,
    diffsense: float = 0.5,
) -> Dict[str, float]:
    """Return S²MATCH precision, recall, and F1 for two AMR graphs."""
    from amr_metrics.s2match_core import score_pair

    amr_a = normalize_amr_line(graph_a)
    amr_b = normalize_amr_line(graph_b)
    if not amr_a or not amr_b:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    vectors = _load_glove_vectors(vectors_path or default_glove_path())
    try:
        precision, recall, f1 = score_pair(
            amr_a,
            amr_b,
            vectors,
            cutoff=cutoff,
            diffsense=diffsense,
        )
    except Exception as exc:
        raise AMRMetricError(f"S²MATCH failed: {exc}") from exc

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def add_amr_similarity_scores(
    amr_df: pd.DataFrame,
    pair_mode: PairMode,
    compute_smatch: bool = True,
    compute_s2match: bool = True,
    vectors_path: Optional[str] = None,
) -> pd.DataFrame:
    """Add per-pair SMATCH and S²MATCH F1 columns to an AMR results dataframe."""
    if pair_mode != "en_en":
        raise AMRMetricError(
            "SMATCH and S²MATCH compare two English AMR graphs per pair. "
            "Use English–English mode, or compare graphs from two English sentences."
        )

    out = amr_df.copy()
    smatch_scores = []
    s2match_scores = []

    for _, row in out.iterrows():
        graph_a = row.get("AMR (A)", "")
        graph_b = row.get("AMR (B)", "")

        if compute_smatch:
            smatch_scores.append(score_smatch_pair(graph_a, graph_b)["f1"])
        if compute_s2match:
            s2match_scores.append(score_s2match_pair(graph_a, graph_b, vectors_path=vectors_path)["f1"])

    if compute_smatch:
        out["score_smatch"] = smatch_scores
    if compute_s2match:
        out["score_s2match"] = s2match_scores

    return out
