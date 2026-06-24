# Sentence Pair Similarity (Streamlit)

Streamlit app for scoring aligned sentence pairs (German–English semi-parallel or English–English paraphrases).

## Run locally

```bash
pip install -r requirements.txt
streamlit run semiparallelapp.py
```

Optional: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your OpenAI API key.

## Deploy to Streamlit Community Cloud

1. **Sync** the latest app code from the parent project (if you edit files there):

   ```bash
   ./sync_from_parent.sh
   ```

2. **Initialize git** in this folder (not the whole research directory):

   ```bash
   git init
   git add .
   git commit -m "Sentence pair similarity Streamlit app"
   ```

3. **Create a GitHub repo** and push:

   ```bash
   gh repo create sentence-pair-similarity --public --source=. --push
   ```

4. **Deploy** at [share.streamlit.io](https://share.streamlit.io):
   - Main file: `semiparallelapp.py`
   - Python: 3.11 or 3.12 (match your local version)
   - **Secrets** (App settings → Secrets):

     ```toml
     OPENAI_API_KEY = "sk-..."
     ```

## What is included vs excluded

| Included | Excluded (too large or local-only) |
|----------|-------------------------------------|
| App Python modules | Research datasets (`EDU:…`, `Misra datasets/`) |
| `arendt_hucon_de-en_sents.csv` (example) | `vectors/glove.6B.100d.txt` (331 MB) |
| `requirements.txt`, `packages.txt` | amrlib `model_stog` weights |

## Cloud limitations

- **Memory:** ~2.7 GB max. Enable only a few scorers at once (LaBSE + char n-gram is safest).
- **Ollama:** Not available on Streamlit Cloud; use OpenAI or disable the LLM judge.
- **S²MATCH:** Needs GloVe vectors hosted elsewhere; set `GLOVE_VECTORS_PATH` in secrets if you host them.
- **AMR parsing:** Requires downloading the [amrlib model_stog](https://github.com/bjascob/amrlib-models/releases) separately (not bundled).
- **AMR graph images:** `packages.txt` installs system Graphviz; first deploy may take 15–20 minutes while PyTorch and Hugging Face models download.

## Updating the app

Edit source in the parent folder, then run `./sync_from_parent.sh`, commit, and push.
