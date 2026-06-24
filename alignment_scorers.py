"""
Alignment scorers from window_align-4.ipynb ("Define all Scorers").
Adapted for pairwise scoring of semi-parallel aligned sentences.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Literal, Optional

PairMode = Literal["de_en", "en_en"]

import numpy as np
import pandas as pd
import torch
from bert_score.scorer import BERTScorer
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class AlignmentScorer:
    def score(self, en_text: str, de_texts: List[str], en_id=None, de_ids=None) -> float:
        raise NotImplementedError


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = _l2_normalize(a)
    b = _l2_normalize(b)
    return float(np.dot(a, b))


def normalize_fit_df(
    de_texts: List[str],
    en_texts: List[str],
    df_for_fit: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """DataFrame with notebook columns Text 1, Text 2, Index for scorer init."""
    n = len(de_texts)
    if df_for_fit is not None and {"Text 1", "Text 2"}.issubset(df_for_fit.columns):
        if len(df_for_fit) == n:
            idx = (
                df_for_fit["Index"].tolist()
                if "Index" in df_for_fit.columns
                else list(range(1, n + 1))
            )
            return pd.DataFrame(
                {
                    "Text 1": df_for_fit["Text 1"].fillna("").astype(str).tolist(),
                    "Text 2": df_for_fit["Text 2"].fillna("").astype(str).tolist(),
                    "Index": idx,
                }
            )

    return pd.DataFrame(
        {
            "Text 1": de_texts,
            "Text 2": en_texts,
            "Index": range(1, n + 1),
        }
    )


class SentenceEmbeddingScorer(AlignmentScorer):
    len_penalty = 0.02

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model = SentenceTransformer(model)
        self.de_id_to_idx = {}
        self.en_vecs = None
        self.de_vecs = None

        if df is not None:
            self.de_texts = df["Text 1"].tolist()
            self.en_texts = df["Text 2"].tolist()
            self.en_vecs = self.model.encode(self.en_texts, normalize_embeddings=True)
            self.de_vecs = self.model.encode(self.de_texts, normalize_embeddings=True)
            self.de_id_to_idx = {d: i for i, d in enumerate(df["Index"])}

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        if en_id is not None and de_ids is not None and self.en_vecs is not None:
            e = self.en_vecs[en_id - 1]
            vecs = [self.de_vecs[self.de_id_to_idx[i]] for i in de_ids]
            d = np.mean(vecs, axis=0)
            return float(np.dot(e, d))

        de = " ".join(de_texts) if de_texts else ""
        e = self.model.encode([en_text], normalize_embeddings=True)[0]
        d = self.model.encode([de], normalize_embeddings=True)[0]
        return float(np.dot(e, d))


class LaBSECosineScorer(AlignmentScorer):
    len_penalty = 0.08

    def __init__(
        self,
        df=None,
        model_name: str = "sentence-transformers/LaBSE",
        device=None,
        batch_size: int = 64,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._cache: Dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        text = "" if text is None else str(text)
        if text in self._cache:
            return self._cache[text]

        v = self.model.encode(
            [text],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)

        self._cache[text] = v
        return v

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        de = " ".join(de_texts) if de_texts else ""
        v_en = self._embed(en_text)
        v_de = self._embed(de)
        return float(np.dot(v_en, v_de))


class BERTScoreScorer(AlignmentScorer):
    len_penalty = 0.00

    def __init__(
        self,
        df=None,
        model: str = "xlm-roberta-large",
        device=None,
        batch_size: int = 32,
        lang: str = "de",
    ):
        self.scorer = BERTScorer(
            model_type=model,
            lang=lang,
            rescale_with_baseline=True,
            device=device,
            batch_size=batch_size,
        )

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        de = " ".join(de_texts)
        _, _, f1 = self.scorer.score([de], [en_text])
        return float(f1[0])

    def score_many(self, en_texts, de_texts):
        de = " ".join(de_texts)
        cands = [de] * len(en_texts)
        _, _, f1 = self.scorer.score(cands, en_texts)
        return [float(x) for x in f1]


class CharNGramScorer(AlignmentScorer):
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        n: int = 3,
        fit_column: str = "Text 1",
    ):
        self.n = n
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), lowercase=True
        )
        self._vec_cache: Dict[str, object] = {}

        if df is not None:
            corpus = df[fit_column].fillna("").astype(str).tolist()
            self.vectorizer.fit(corpus)

    def fit_german_texts(self, texts: List[str]) -> None:
        self.vectorizer.fit([str(t) for t in texts])
        self._vec_cache.clear()

    def _get_vector(self, text):
        if text not in self._vec_cache:
            self._vec_cache[text] = self.vectorizer.transform([text])
        return self._vec_cache[text]

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        de = " ".join(de_texts)
        v_en = self._get_vector(en_text)
        v_de = self._get_vector(de)
        return float(cosine_similarity(v_en, v_de)[0][0])


class NLIScorer(AlignmentScorer):
    len_penalty = 0.1

    def __init__(
        self,
        df=None,
        model: str = "joeddav/xlm-roberta-large-xnli",
        device=None,
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

        label2id = getattr(self.model.config, "label2id", {}) or {}
        label2id_norm = {str(k).lower(): int(v) for k, v in label2id.items()} if label2id else {}
        self.entailment_id = label2id_norm.get("entailment", 2)

    @torch.inference_mode()
    def _entailment_prob(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self.entailment_id].item())

    @torch.inference_mode()
    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        text_a = "" if en_text is None else str(en_text)
        text_b = " ".join(de_texts) if de_texts else ""
        forward = self._entailment_prob(text_a, text_b)
        backward = self._entailment_prob(text_b, text_a)
        return (forward + backward) / 2.0

    @torch.inference_mode()
    def score_many(self, en_texts, de_texts):
        de = " ".join(de_texts) if de_texts else ""
        if not en_texts:
            return []

        forward_enc = self.tokenizer(
            en_texts,
            [de] * len(en_texts),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        forward_enc = {k: v.to(self.device) for k, v in forward_enc.items()}
        forward_probs = torch.softmax(self.model(**forward_enc).logits, dim=-1)[:, self.entailment_id]

        backward_enc = self.tokenizer(
            [de] * len(en_texts),
            en_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        backward_enc = {k: v.to(self.device) for k, v in backward_enc.items()}
        backward_probs = torch.softmax(self.model(**backward_enc).logits, dim=-1)[:, self.entailment_id]

        return [
            float((f + b) / 2.0)
            for f, b in zip(forward_probs.detach().cpu().tolist(), backward_probs.detach().cpu().tolist())
        ]


class APTParaphraseScorer(AlignmentScorer):
    """Paraphrase probability from Nighojkar & Licato (2021), arXiv:2106.07691."""

    len_penalty = 0.0

    def __init__(
        self,
        df=None,
        model: str = "AMHR/adversarial-paraphrasing-detector",
        device=None,
        max_length: int = 256,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

        label2id = getattr(self.model.config, "label2id", {}) or {}
        label2id_norm = {str(k).lower(): int(v) for k, v in label2id.items()} if label2id else {}
        self.paraphrase_id = label2id_norm.get("label_1", 1)

    @torch.inference_mode()
    def _paraphrase_prob(self, sentence_a: str, sentence_b: str) -> float:
        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, self.paraphrase_id].item())

    @torch.inference_mode()
    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        text_a = "" if en_text is None else str(en_text)
        text_b = " ".join(de_texts) if de_texts else ""
        forward = self._paraphrase_prob(text_a, text_b)
        backward = self._paraphrase_prob(text_b, text_a)
        return (forward + backward) / 2.0


class JinaRerankerScorer(AlignmentScorer):
    """Cross-lingual relevance via jinaai/jina-reranker-v3 (EN query → DE document)."""

    len_penalty = 0.05

    def __init__(
        self,
        df=None,
        model: str = "jinaai/jina-reranker-v3",
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        load_kwargs = {"trust_remote_code": True}
        try:
            self.model = AutoModel.from_pretrained(model, dtype="auto", **load_kwargs)
        except TypeError:
            self.model = AutoModel.from_pretrained(
                model, torch_dtype="auto", **load_kwargs
            )
        self.model.to(self.device)
        self.model.eval()

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        en_text = "" if en_text is None else str(en_text)
        de = " ".join(de_texts) if de_texts else ""
        if not en_text.strip() and not de.strip():
            return 0.0

        with torch.inference_mode():
            results = self.model.rerank(en_text, [de])

        if not results:
            return 0.0
        by_index = {int(r["index"]): float(r["relevance_score"]) for r in results}
        return by_index.get(0, float(results[0]["relevance_score"]))


class LLMScorerError(Exception):
    """User-facing LLM scorer failure (quota, auth, connectivity)."""


class LLMScorer(AlignmentScorer):
    """Semantic similarity via an OpenAI-compatible chat model (0–1 rating)."""

    len_penalty = 0.0

    _SYSTEM = (
        "You rate semantic similarity between sentence pairs for academic NLP research. "
        "Always reply with exactly one number from 0.0 to 1.0. "
        "Use low scores when meanings differ. Never refuse or explain."
    )

    _PROMPTS = {
        "de_en": (
            "Rate how equivalent in meaning the German and English sentences are.\n\n"
            "German: {left}\n"
            "English: {right}\n\n"
            "0.0 = unrelated meanings, 1.0 = fully equivalent. Reply with one number only."
        ),
        "en_en": (
            "Rate how similar in meaning the two English sentences are.\n\n"
            "Sentence A: {left}\n"
            "Sentence B: {right}\n\n"
            "0.0 = unrelated meanings, 1.0 = same meaning. Reply with one number only."
        ),
    }

    _REFUSAL_RE = re.compile(
        r"\b(cannot|can't|unable to|will not|won't|not able to|i refuse)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        df=None,
        pair_mode: PairMode = "de_en",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key and base_url:
            key = "ollama"
        if not key:
            raise ValueError(
                "LLM scorer requires an API key. Set OPENAI_API_KEY or enter a key in the sidebar."
            )

        client_kwargs: Dict[str, Any] = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.pair_mode = pair_mode
        self.base_url = base_url

    def _parse_score(self, text: str) -> float:
        text = text.strip()
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            value = float(match.group(1))
            if value > 1.0 and value <= 100.0:
                value /= 100.0
            return max(0.0, min(1.0, value))
        if self._REFUSAL_RE.search(text):
            return 0.0
        raise LLMScorerError(
            f"LLM returned non-numeric text instead of a score: {text!r}. "
            "Try a different model in the sidebar, or disable the LLM judge."
        )

    def _request_score(self, left: str, right: str) -> str:
        prompt = self._PROMPTS[self.pair_mode].format(left=left, right=right)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16,
        )
        return (response.choices[0].message.content or "").strip()

    def _quota_help(self) -> str:
        if self.base_url and "11434" in self.base_url:
            return (
                "Check that Ollama is running (`ollama serve`) and the model is pulled "
                f"(`ollama pull {self.model}`)."
            )
        return (
            "Your OpenAI account has no remaining quota. Add billing at "
            "https://platform.openai.com/account/billing, or switch the sidebar "
            "provider to **Ollama (local)** to run a free model on your machine."
        )

    def score(self, en_text, de_texts, en_id=None, de_ids=None) -> float:
        from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError

        left = " ".join(de_texts) if de_texts else ""
        right = "" if en_text is None else str(en_text)

        def _call_with_retry() -> float:
            content = self._request_score(left, right)
            try:
                return self._parse_score(content)
            except LLMScorerError:
                retry_prompt = (
                    f"{self._PROMPTS[self.pair_mode].format(left=left, right=right)}\n\n"
                    "Respond with only a number between 0.0 and 1.0."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._SYSTEM},
                        {"role": "user", "content": retry_prompt},
                    ],
                    temperature=0,
                    max_tokens=8,
                )
                return self._parse_score(response.choices[0].message.content or "")

        try:
            return _call_with_retry()
        except LLMScorerError:
            raise
        except RateLimitError as exc:
            body = str(exc).lower()
            if "insufficient_quota" in body or "exceeded your current quota" in body:
                raise LLMScorerError(self._quota_help()) from exc
            raise LLMScorerError(
                f"LLM rate limit reached. Wait a moment and retry, or use fewer pairs. ({exc})"
            ) from exc
        except AuthenticationError as exc:
            raise LLMScorerError(
                "Invalid API key. Check the key in the sidebar or OPENAI_API_KEY."
            ) from exc
        except APIConnectionError as exc:
            raise LLMScorerError(
                f"Could not reach the LLM API at {self.base_url or 'api.openai.com'}. "
                "Is the server running?"
            ) from exc
        except APIError as exc:
            raise LLMScorerError(f"LLM API error: {exc}") from exc


SCORER_LABELS_DE_EN = {
    "cosine": "Multilingual MiniLM (cosine)",
    "labse": "LaBSE (cosine)",
    "bertscore": "BERTScore (XLM-R, F1)",
    "charngram": "Character n-gram TF-IDF",
    "nli": "XNLI entailment (bidirectional average)",
    "jina": "Jina reranker v3 (EN query → DE relevance)",
    "llm": "LLM judge (OpenAI-compatible)",
}

SCORER_LABELS_EN_EN = {
    "cosine": "Multilingual MiniLM (cosine)",
    "labse": "LaBSE (cosine)",
    "bertscore": "BERTScore (XLM-R, F1)",
    "charngram": "Character n-gram TF-IDF",
    "nli": "XNLI entailment (bidirectional average)",
    "jina": "Jina reranker v3 (A query → B relevance)",
    "apt_pd": "APT paraphrase detector (Nighojkar & Licato 2021)",
    "llm": "LLM judge (OpenAI-compatible)",
}

SCORER_LABELS = SCORER_LABELS_DE_EN


def scorer_labels_for_mode(pair_mode: PairMode = "de_en") -> Dict[str, str]:
    if pair_mode == "en_en":
        return SCORER_LABELS_EN_EN
    return SCORER_LABELS_DE_EN


def make_scorer(
    kind: str,
    df: Optional[pd.DataFrame] = None,
    pair_mode: PairMode = "de_en",
    scorer_options: Optional[Dict[str, Dict[str, Any]]] = None,
) -> AlignmentScorer:
    bert_lang = "en" if pair_mode == "en_en" else "de"
    char_fit_column = "Text 1"
    llm_opts = (scorer_options or {}).get("llm", {})
    scorers = {
        "cosine": lambda: SentenceEmbeddingScorer(df),
        "labse": lambda: LaBSECosineScorer(df),
        "bertscore": lambda: BERTScoreScorer(df, lang=bert_lang),
        "charngram": lambda: CharNGramScorer(df, fit_column=char_fit_column),
        "nli": lambda: NLIScorer(df),
        "jina": lambda: JinaRerankerScorer(df),
        "apt_pd": lambda: APTParaphraseScorer(df),
        "llm": lambda: LLMScorer(df, pair_mode=pair_mode, **llm_opts),
    }
    if kind not in scorers:
        raise ValueError(f"Unknown scorer kind: {kind}")
    return scorers[kind]()


def score_aligned_pairs(
    texts_1: List[str],
    texts_2: List[str],
    scorer_kinds: List[str],
    df_for_fit: Optional[pd.DataFrame] = None,
    progress_callback=None,
    pair_mode: PairMode = "de_en",
    scorer_options: Optional[Dict[str, Dict[str, Any]]] = None,
    scorer_cache: Optional[Dict[tuple, AlignmentScorer]] = None,
) -> pd.DataFrame:
    """Score each aligned sentence pair with the selected scorers."""
    if len(texts_1) != len(texts_2):
        raise ValueError("Both sides of each pair must have the same length.")

    labels = scorer_labels_for_mode(pair_mode)
    n = len(texts_1)
    if pair_mode == "de_en":
        out = pd.DataFrame({"Text 1 (DE)": texts_1, "Text 2 (EN)": texts_2})
    else:
        out = pd.DataFrame({"Sentence A": texts_1, "Sentence B": texts_2})

    fit_df = normalize_fit_df(texts_1, texts_2, df_for_fit)

    cache = scorer_cache if scorer_cache is not None else {}

    for kind in scorer_kinds:
        if progress_callback:
            progress_callback(f"Loading {labels[kind]}…")
        if kind == "llm":
            scorer = make_scorer(kind, fit_df, pair_mode=pair_mode, scorer_options=scorer_options)
        else:
            cache_key = (kind, pair_mode)
            if cache_key not in cache:
                cache[cache_key] = make_scorer(
                    kind, fit_df, pair_mode=pair_mode, scorer_options=scorer_options
                )
            scorer = cache[cache_key]
        col = f"score_{kind}"
        scores: List[float] = []

        for i in range(n):
            left = str(texts_1[i]) if texts_1[i] is not None else ""
            right = str(texts_2[i]) if texts_2[i] is not None else ""
            if pair_mode == "de_en":
                query, doc = right, left
            else:
                query, doc = left, right
            scores.append(scorer.score(query, [doc]))
            if progress_callback and (i + 1) % max(1, n // 10) == 0:
                progress_callback(f"{labels[kind]}: {i + 1}/{n}")

        out[col] = scores

    return out
