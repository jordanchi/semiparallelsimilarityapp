"""
Streamlit app: similarity scores for aligned sentence pairs
using the scorers defined in window_align-4.ipynb.
"""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st

from alignment_scorers import (
    LLMScorerError,
    PairMode,
    scorer_labels_for_mode,
    score_aligned_pairs,
)
from amr_utils import AMRParserError, build_amr_dataframe, render_amr_graph_png
from amr_metrics import AMRMetricError, add_amr_similarity_scores, default_glove_path


def _openai_api_key() -> str:
    """Read OpenAI key from Streamlit secrets (cloud) or environment (local)."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return str(key)
    except (FileNotFoundError, AttributeError, KeyError):
        pass
    return os.environ.get("OPENAI_API_KEY", "")


st.set_page_config(
    page_title="Sentence Pair Similarity",
    page_icon="↔️",
    layout="wide",
)

DE_EN_SAMPLE = """Die Menschen, die Welt, die Erde und das All - davon ist in diesem Buch ausdrücklich nicht die Rede.
In 1957, an earth-born object made by man was launched into the universe, where for some weeks it circled the earth according to the same laws of gravitation that swing and keep in motion the celestial bodies—the sun, the moon, and the stars.
Audi nicht davon, wie die von Menschen errichtete Welt von der Erde weg in den Himmel sich streckt.
To be sure, the man-made satellite was no moon or star, no heavenly body which could follow its circling path for a time span that to us mortals, bound by earthly time, lasts from eternity to eternity."""

EN_EN_SAMPLE = """The cat sat on the mat. | A feline rested on the rug.
She opened the window to let in fresh air. | She opened the window so fresh air could come in.
The experiment failed because the sample was contaminated. | The trial did not succeed due to contamination of the sample.
He refused to comment on the allegations. | He declined to make any statement about the accusations."""

EN_EN_BUILTIN = [
    (
        "The cat sat on the mat.",
        "A feline rested on the rug.",
    ),
    (
        "She opened the window to let in fresh air.",
        "She opened the window so fresh air could come in.",
    ),
    (
        "The experiment failed because the sample was contaminated.",
        "The trial did not succeed due to contamination of the sample.",
    ),
    (
        "He refused to comment on the allegations.",
        "He declined to make any statement about the accusations.",
    ),
    (
        "The committee will meet again next week.",
        "The panel is scheduled to convene again the following week.",
    ),
    (
        "Traffic was delayed by an accident on the highway.",
        "A crash on the motorway caused long delays for drivers.",
    ),
    (
        "The novel explores themes of exile and belonging.",
        "The book examines ideas of displacement and home.",
    ),
    (
        "Investors reacted cautiously to the earnings report.",
        "Shareholders responded with caution to the quarterly results.",
    ),
]

pair_mode: PairMode = st.radio(
    "Pair type",
    options=["de_en", "en_en"],
    format_func=lambda x: (
        "German–English (semi-parallel)"
        if x == "de_en"
        else "English–English (paraphrases)"
    ),
    horizontal=True,
)

is_de_en = pair_mode == "de_en"
scorer_labels = scorer_labels_for_mode(pair_mode)

if is_de_en:
    st.title("Semi-parallel sentence similarity")
    st.caption(
        "Score aligned German–English sentence pairs with the alignment metrics "
        "from `window_align-4.ipynb` (Define all Scorers)."
    )
else:
    st.title("English paraphrase similarity")
    st.caption(
        "Score aligned English–English sentence pairs (e.g. paraphrases) "
        "with the same alignment metrics."
    )

with st.sidebar:
    if is_de_en:
        st.info(
            "**APT** and **AMR SMATCH** require **English–English** pair type "
            "(switch at the top of the page)."
        )

    st.header("Scorers")
    selected_scorers = [
        key
        for key, label in scorer_labels.items()
        if st.checkbox(label, value=key in ("labse", "bertscore", "charngram"))
    ]
    if not is_de_en:
        st.caption("Includes **APT paraphrase detector** — enable it above, then click **Compute similarity scores**.")

    llm_provider = "OpenAI"
    llm_api_key = _openai_api_key()
    llm_model = "gpt-4o-mini"
    llm_base_url = ""
    if "llm" in selected_scorers:
        st.divider()
        st.subheader("LLM settings")
        llm_provider = st.selectbox(
            "Provider",
            ["OpenAI", "Ollama (local)", "Custom"],
            help="Ollama runs free models on your machine; install from https://ollama.com",
        )
        if llm_provider == "Ollama (local)":
            llm_base_url = "http://localhost:11434/v1"
            llm_model = "llama3.2"
            llm_api_key = "ollama"
            st.caption(
                "Requires Ollama running locally. In a terminal: "
                "`ollama pull llama3.2` then `ollama serve`."
            )
            llm_model = st.text_input("Ollama model", value=llm_model)
        elif llm_provider == "OpenAI":
            llm_api_key = st.text_input(
                "API key",
                value=llm_api_key,
                type="password",
                help="Requires billing enabled at platform.openai.com.",
            )
            llm_model = st.text_input("Model", value=llm_model)
        else:
            llm_api_key = st.text_input(
                "API key",
                value=llm_api_key,
                type="password",
            )
            llm_model = st.text_input("Model", value=llm_model)
            llm_base_url = st.text_input(
                "Base URL",
                value="",
                placeholder="https://api.openai.com/v1",
            )

    st.divider()
    st.header("AMR")
    show_amr = st.checkbox(
        "Parse AMR graphs",
        help="Abstract Meaning Representation via amrlib. English sentences only.",
    )
    show_amr_graphs = False
    compute_amr_smatch = False
    compute_amr_s2match = False
    glove_vectors_path = default_glove_path()
    if not is_de_en:
        st.caption("SMATCH / S²MATCH compare the two English AMR graphs in each pair.")
        compute_amr_smatch = st.checkbox("Compute SMATCH", value=True)
        compute_amr_s2match = st.checkbox(
            "Compute S²MATCH",
            value=False,
            help="Requires GloVe vectors (glove.6B.100d.txt).",
        )
        if compute_amr_s2match:
            glove_vectors_path = st.text_input(
                "GloVe vectors path",
                value=default_glove_path(),
                help="Set GLOVE_VECTORS_PATH or place glove.6B.100d.txt in `vectors/`.",
            )
    if show_amr:
        show_amr_graphs = st.checkbox(
            "Show graph images",
            help="Requires the Graphviz system package (e.g. `brew install graphviz`).",
        )
        if is_de_en:
            st.caption("German sentences are skipped; only the English side is parsed.")

    st.divider()
    st.markdown(
        "**Note:** First run downloads model weights. "
        "BERTScore, NLI, and Jina reranker v3 are slower. "
        "The LLM judge calls an API once per sentence pair. "
        "AMR parsing requires a separate amrlib model download."
    )

if is_de_en:
    input_options = ["Upload CSV", "Paste aligned lines", "Example (Arendt sample)"]
else:
    input_options = ["Upload CSV", "Paste aligned lines", "Example (paraphrases)"]

input_mode = st.radio("Input mode", input_options, horizontal=True)

texts_1: list[str] = []
texts_2: list[str] = []
source_df: pd.DataFrame | None = None

col_1_label = "German" if is_de_en else "Sentence A"
col_2_label = "English" if is_de_en else "Sentence B"

if input_mode == "Upload CSV":
    upload_label = (
        "CSV with aligned DE/EN columns"
        if is_de_en
        else "CSV with aligned English sentence columns"
    )
    uploaded = st.file_uploader(upload_label, type=["csv"])
    if uploaded:
        source_df = pd.read_csv(uploaded)
        cols = list(source_df.columns)
        default_1 = "Text 1" if "Text 1" in cols else cols[0]
        default_2 = "Text 2" if "Text 2" in cols else cols[min(1, len(cols) - 1)]

        col_a, col_b = st.columns(2)
        with col_a:
            col_1 = st.selectbox(f"{col_1_label} column", cols, index=cols.index(default_1))
        with col_b:
            col_2 = st.selectbox(f"{col_2_label} column", cols, index=cols.index(default_2))

        texts_1 = source_df[col_1].fillna("").astype(str).tolist()
        texts_2 = source_df[col_2].fillna("").astype(str).tolist()
        st.dataframe(source_df[[col_1, col_2]].head(8), use_container_width=True)

elif input_mode == "Paste aligned lines":
    if is_de_en:
        paste_help = (
            "Enter **one pair per row**: German sentence, then English sentence, "
            "separated by a tab or ` | `. Blank lines are skipped."
        )
        placeholder = DE_EN_SAMPLE
    else:
        paste_help = (
            "Enter **one pair per row**: sentence A, then sentence B, "
            "separated by a tab or ` | `. Blank lines are skipped."
        )
        placeholder = EN_EN_SAMPLE

    st.markdown(paste_help)
    pasted = st.text_area("Aligned pairs", height=220, placeholder=placeholder)
    for line in pasted.splitlines():
        line = line.strip()
        if not line:
            continue
        if "\t" in line:
            parts = line.split("\t", 1)
        elif " | " in line:
            parts = line.split(" | ", 1)
        else:
            st.warning(f"Skipped (no tab or ' | '): {line[:60]}…")
            continue
        texts_1.append(parts[0].strip())
        texts_2.append(parts[1].strip())

elif is_de_en:
    sample_path = "arendt_hucon_de-en_sents.csv"
    try:
        source_df = pd.read_csv(sample_path)
        texts_1 = source_df["Text 1"].fillna("").astype(str).tolist()
        texts_2 = source_df["Text 2"].fillna("").astype(str).tolist()
        max_rows = st.slider("Number of example rows", 2, min(30, len(texts_1)), 10)
        texts_1 = texts_1[:max_rows]
        texts_2 = texts_2[:max_rows]
        st.info(f"Loaded {len(texts_1)} pairs from `{sample_path}`.")
    except FileNotFoundError:
        st.error(f"Place `{sample_path}` next to this app, or use another input mode.")
else:
    max_rows = st.slider(
        "Number of example rows",
        2,
        len(EN_EN_BUILTIN),
        min(4, len(EN_EN_BUILTIN)),
    )
    texts_1 = [pair[0] for pair in EN_EN_BUILTIN[:max_rows]]
    texts_2 = [pair[1] for pair in EN_EN_BUILTIN[:max_rows]]
    st.info(f"Loaded {len(texts_1)} built-in English paraphrase pairs.")

if not texts_1:
    st.stop()

if len(texts_1) != len(texts_2):
    st.error(
        f"Row count mismatch: {len(texts_1)} {col_1_label} vs {len(texts_2)} {col_2_label}."
    )
    st.stop()

st.subheader(f"{len(texts_1)} aligned pairs ready")

action_col1, action_col2 = st.columns(2)
with action_col1:
    run = st.button(
        "Compute similarity scores",
        type="primary",
        disabled=not selected_scorers,
    )
with action_col2:
    run_amr = st.button(
        "Parse AMR graphs",
        disabled=not show_amr,
    )

if not selected_scorers and not show_amr:
    st.info("Select scorers and/or enable AMR parsing in the sidebar.")
elif not selected_scorers:
    st.caption("Similarity scoring is disabled until you select at least one scorer.")
elif not show_amr:
    st.caption("Enable **Parse AMR graphs** in the sidebar to parse AMR.")

if run:
    if (
        "llm" in selected_scorers
        and llm_provider == "OpenAI"
        and not (llm_api_key or _openai_api_key())
    ):
        st.error(
            "LLM scorer selected: enter an OpenAI API key in the sidebar, "
            "set OPENAI_API_KEY in Streamlit secrets, or export OPENAI_API_KEY locally."
        )
        st.stop()

    status = st.empty()
    progress = st.progress(0.0)

    def on_progress(msg: str):
        status.info(msg)

    scorer_options = {
        "llm": {
            "api_key": llm_api_key or None,
            "model": llm_model,
            "base_url": llm_base_url or None,
        }
    }

    try:
        if is_de_en:
            results = pd.DataFrame({"Text 1 (DE)": texts_1, "Text 2 (EN)": texts_2})
        else:
            results = pd.DataFrame({"Sentence A": texts_1, "Sentence B": texts_2})

        for idx, kind in enumerate(selected_scorers):
            on_progress(f"Running {scorer_labels[kind]}…")
            progress.progress((idx) / len(selected_scorers))

            if "scorer_cache" not in st.session_state:
                st.session_state["scorer_cache"] = {}

            partial = score_aligned_pairs(
                texts_1,
                texts_2,
                [kind],
                df_for_fit=source_df,
                pair_mode=pair_mode,
                scorer_options=scorer_options,
                scorer_cache=st.session_state["scorer_cache"],
            )
            results[f"score_{kind}"] = partial[f"score_{kind}"]

        progress.progress(1.0)
        status.success("Done.")

        st.session_state["similarity_results"] = results
        st.session_state["results_pair_mode"] = pair_mode
    except LLMScorerError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.exception(exc)
        st.stop()

if run_amr:
    status = st.empty()
    status.info("Loading amrlib parser and parsing sentences…")
    try:
        amr_results = build_amr_dataframe(texts_1, texts_2, pair_mode)
        if pair_mode == "en_en" and (compute_amr_smatch or compute_amr_s2match):
            status.info("Computing AMR similarity metrics…")
            amr_results = add_amr_similarity_scores(
                amr_results,
                pair_mode,
                compute_smatch=compute_amr_smatch,
                compute_s2match=compute_amr_s2match,
                vectors_path=glove_vectors_path or None,
            )
        st.session_state["amr_results"] = amr_results
        st.session_state["amr_pair_mode"] = pair_mode
        st.session_state["amr_show_graphs"] = show_amr_graphs
        st.session_state["compute_amr_smatch"] = compute_amr_smatch
        st.session_state["compute_amr_s2match"] = compute_amr_s2match
        st.session_state["glove_vectors_path"] = glove_vectors_path
        status.success(f"Parsed AMR for {len(amr_results)} pair(s).")
    except AMRParserError as exc:
        st.error(str(exc))
    except AMRMetricError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

if "amr_results" in st.session_state:
    amr_results = st.session_state["amr_results"]
    amr_pair_mode = st.session_state.get("amr_pair_mode", "de_en")
    amr_show_graphs = st.session_state.get("amr_show_graphs", False)

    st.subheader("AMR representations")
    if amr_pair_mode == "de_en":
        st.caption("AMR graphs for the English side of each pair (German not supported by AMR).")
    else:
        st.caption("AMR graphs for both sentences in each pair.")

    amr_score_cols = [c for c in amr_results.columns if c.startswith("score_")]
    if amr_pair_mode == "en_en" and not amr_score_cols:
        st.warning(
            "No AMR similarity scores yet. Enable **Compute SMATCH** / **Compute S²MATCH** "
            "in the sidebar (AMR section), then click **Parse AMR graphs** again — "
            "or use the button below on already-parsed graphs."
        )
        if st.button("Compute AMR similarity (SMATCH / S²MATCH)"):
            do_smatch = st.session_state.get("compute_amr_smatch", compute_amr_smatch)
            do_s2match = st.session_state.get("compute_amr_s2match", compute_amr_s2match)
            glove_path = st.session_state.get("glove_vectors_path", glove_vectors_path)
            if not (do_smatch or do_s2match):
                st.error("Enable **Compute SMATCH** and/or **Compute S²MATCH** in the sidebar first.")
            else:
                try:
                    amr_results = add_amr_similarity_scores(
                        amr_results,
                        amr_pair_mode,
                        compute_smatch=do_smatch,
                        compute_s2match=do_s2match,
                        vectors_path=glove_path or None,
                    )
                    st.session_state["amr_results"] = amr_results
                    st.rerun()
                except AMRMetricError as exc:
                    st.error(str(exc))

    if amr_score_cols:
        st.subheader("AMR similarity")
        metric_summary = pd.DataFrame(
            {
                "Metric": ["SMATCH" if c == "score_smatch" else "S²MATCH" for c in amr_score_cols],
                "Mean": [amr_results[c].mean() for c in amr_score_cols],
                "Median": [amr_results[c].median() for c in amr_score_cols],
                "Pairs": [amr_results[c].notna().sum() for c in amr_score_cols],
            }
        )
        st.dataframe(
            metric_summary.style.format({"Mean": "{:.4f}", "Median": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            amr_results[
                ["Pair", "Sentence A", "Sentence B", *amr_score_cols]
            ].style.background_gradient(subset=amr_score_cols, cmap="RdYlGn"),
            use_container_width=True,
        )

    for _, row in amr_results.iterrows():
        pair_num = int(row["Pair"])
        with st.expander(f"Pair {pair_num}", expanded=pair_num == 1):
            if amr_pair_mode == "de_en":
                st.markdown(f"**German:** {row['German']}")
                st.markdown(f"**English:** {row['English']}")
                graph = row["AMR (English)"]
                if graph:
                    st.code(graph, language=None)
                    if amr_show_graphs:
                        png = render_amr_graph_png(graph)
                        if png:
                            st.image(png, caption=f"Pair {pair_num} AMR graph")
                        else:
                            st.caption("Graph image unavailable (install Graphviz system package).")
                else:
                    st.caption("No AMR (empty English sentence).")
            else:
                st.markdown(f"**Sentence A:** {row['Sentence A']}")
                graph_a = row["AMR (A)"]
                if graph_a:
                    st.markdown("**AMR (A)**")
                    st.code(graph_a, language=None)
                    if amr_show_graphs:
                        png = render_amr_graph_png(graph_a)
                        if png:
                            st.image(png, caption=f"Pair {pair_num} · Sentence A")
                st.markdown(f"**Sentence B:** {row['Sentence B']}")
                graph_b = row["AMR (B)"]
                if graph_b:
                    st.markdown("**AMR (B)**")
                    st.code(graph_b, language=None)
                    if amr_show_graphs:
                        png = render_amr_graph_png(graph_b)
                        if png:
                            st.image(png, caption=f"Pair {pair_num} · Sentence B")

    amr_buf = io.StringIO()
    amr_results.to_csv(amr_buf, index=False)
    st.download_button(
        "Download AMR results (CSV)",
        amr_buf.getvalue(),
        file_name="amr_graphs.csv",
        mime="text/csv",
    )

if "similarity_results" in st.session_state:
    results = st.session_state["similarity_results"]
    score_cols = [c for c in results.columns if c.startswith("score_")]
    results_labels = scorer_labels_for_mode(
        st.session_state.get("results_pair_mode", "de_en")
    )

    show_summary = st.checkbox("Show mean & median per scorer", value=True)

    if show_summary and score_cols:
        summary = pd.DataFrame(
            {
                "Scorer": [
                    results_labels.get(col.removeprefix("score_"), col) for col in score_cols
                ],
                "Mean": [results[col].mean() for col in score_cols],
                "Median": [results[col].median() for col in score_cols],
                "Pairs": [results[col].notna().sum() for col in score_cols],
            }
        )
        st.subheader("Summary statistics")
        st.dataframe(
            summary.style.format({"Mean": "{:.4f}", "Median": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Scores")
    st.dataframe(
        results.style.background_gradient(subset=score_cols, cmap="RdYlGn"),
        use_container_width=True,
        height=min(520, 35 * len(results) + 38),
    )

    if len(score_cols) > 1:
        st.subheader("Score distributions")
        st.line_chart(results[score_cols])

    buf = io.StringIO()
    results.to_csv(buf, index=False)
    download_name = (
        "semiparallel_similarity.csv"
        if st.session_state.get("results_pair_mode", "de_en") == "de_en"
        else "en_en_paraphrase_similarity.csv"
    )
    st.download_button(
        "Download results (CSV)",
        buf.getvalue(),
        file_name=download_name,
        mime="text/csv",
    )
