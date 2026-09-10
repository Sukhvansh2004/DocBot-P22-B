# Flask / LangChain prototype (archived)

This branch preserves the **earlier Flask-based line of DocBot** that was never
merged into `main`.

`main` is the version that shipped: Streamlit + Sentence-Transformers/Annoy +
Gemma, in `streamlit.py` / `Backend.py` / `Backend2.py`. This branch is a
*different architecture* for the same product — a Flask JSON API with a
hand-written HTML/CSS/JS chat frontend — and it predates `main`.

Nothing here is maintained. It is kept so the approach isn't lost.

## `flask_langchain/` — the API server, four iterations

All four expose a Flask app with roughly the same routes (`/`, `/send`,
`/refresh`, `/upload`, `/get_initial_pdf`) and differ in the retrieval and
generation stack:

| File | Retrieval | Generation |
|---|---|---|
| `app.py` | LangChain `RetrievalQA` over a FAISS index, `PyMuPDFLoader` + `RecursiveCharacterTextSplitter`, BGE embeddings | `CTransformers` running `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` locally |
| `app2.py` | Annoy index over Sentence-Transformers `multi-qa-mpnet-base-cos-v1`; LangChain dropped | `transformers` pipeline |
| `app3.py` | Same as `app2`, plus a **Redis** cache for the serialised Annoy index (`localhost:6379`) | `meta-llama/Meta-Llama-3-8B` text-generation pipeline |
| `app4.py` | Same Annoy retrieval | `AutoModelForCausalLM` with explicit device placement |
| `Untitled-1.py` | — | A 23-line scratch fragment; superseded by `app3.py` |

`app.py` also has a `load_japanese_models()` stub with placeholder paths
(`your_japanese_model_path`) — a multilingual branch that was never filled in.
`test.ipynb` is a scratch notebook.

Note the direction of travel: the first version leaned on LangChain and a local
GGUF Mistral, then LangChain and FAISS were dropped in favour of a direct
Annoy + Sentence-Transformers index, which is the retrieval approach that
survives in `main`'s `Backend.py`.

## `web/` — the HTML frontend

- `chatbot.html`, `chatbot2.html` — the chat UI served by the Flask app, with
  `scripts/script.js` and `styles/style.css`.
- `chatbot2_standalone.html`, `sample_html.html` — self-contained snapshots with
  the assets inlined.
- `backend.py`, `test.py` — the server-side pieces that sat alongside the page.

`main` has no HTML at all; Streamlit renders the UI there.

## Running it

Not recommended without work, but if you want to:

```shell
pip install flask langchain sentence-transformers annoy transformers torch \
            pymupdf pdfbox faiss-cpu redis
export HF_TOKEN=<your token>     # required: the models are gated
python flask_langchain/app.py    # or app2 / app3 / app4
```

`app3.py` additionally needs a Redis server on `localhost:6379`. The Llama-3 and
Gemma models are gated on the Hub, so the token must belong to an account that
has accepted their licences.

## Credentials

These files originally had a Hugging Face access token hard-coded as
`auth_token = "hf_..."` in `app2.py`, `app3.py`, `app4.py`, `Untitled-1.py`,
`web/backend.py` and `web/test.py`. Those six lines now read
`os.environ.get("HF_TOKEN")` instead.

That same token — and a second one — are still committed on `main` in
`Backend.py:14` and `Backend2.py:14`. **Both should be revoked** at
<https://huggingface.co/settings/tokens>: the repository is public and has been
forked, so the values exist in copies outside this repo and removing the lines
is not sufficient.
