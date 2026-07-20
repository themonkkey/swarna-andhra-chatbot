"""Embedding provider abstraction for the Swarna Andhra RAG index.

Swappable by env var EMBED_PROVIDER, exactly like the LLM in app.py:
  - "cohere" (default): embed-english-v3.0, no-card trial key, high RPM. Needs COHERE_API_KEY.
  - "voyage": voyage-3.5-lite, generous free tokens but only 3 RPM without a card. Needs VOYAGE_API_KEY.
  - "gemini": Google gemini-embedding-001, free but ~1000/day cap. Needs GEMINI_API_KEY.
  - "openai": text-embedding-3-small, cheap + high quality. Needs OPENAI_API_KEY.

Both return L2-normalized float32 vectors so cosine similarity is a plain dot product.
The index MUST be built and queried with the same provider + model (vector spaces
are not interchangeable) — the model name is stamped into the saved index and checked
at load time.
"""
import os

import numpy as np

COHERE_MODEL = os.environ.get("EMBED_MODEL", "embed-english-v3.0")
VOYAGE_MODEL = os.environ.get("EMBED_MODEL", "voyage-3.5-lite")
GEMINI_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
OPENAI_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
# gemini-embedding-001 is natively 3072-dim; Matryoshka truncation to 768 keeps
# retrieval quality while shrinking the stored matrix ~4x (deploy-friendly RAM).
GEMINI_DIM = int(os.environ.get("EMBED_DIM", "768"))
# task types let Gemini optimize doc-vs-query embeddings differently
_GEMINI_DOC_TASK = "RETRIEVAL_DOCUMENT"
_GEMINI_QUERY_TASK = "RETRIEVAL_QUERY"


def provider():
    return os.environ.get("EMBED_PROVIDER", "cohere").lower()


def model_id():
    p = provider()
    if p == "cohere":
        return f"cohere:{COHERE_MODEL}"
    if p == "voyage":
        return f"voyage:{VOYAGE_MODEL}"
    if p == "gemini":
        return f"gemini:{GEMINI_MODEL}"
    if p == "openai":
        return f"openai:{OPENAI_MODEL}"
    raise RuntimeError(f"Unknown EMBED_PROVIDER: {p}")


def _normalize(mat):
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _embed_cohere(texts, is_query, api_key=None):
    import requests

    key = api_key or os.environ["COHERE_API_KEY"]
    r = requests.post(
        "https://api.cohere.com/v2/embed",
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        json={"texts": texts, "model": COHERE_MODEL,
              "input_type": "search_query" if is_query else "search_document",
              "embedding_types": ["float"]},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"cohere {r.status_code}: {r.text[:200]}")
    return r.json()["embeddings"]["float"]


def _embed_voyage(texts, is_query):
    import requests

    r = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {os.environ['VOYAGE_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"input": texts, "model": VOYAGE_MODEL,
              "input_type": "query" if is_query else "document"},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"voyage {r.status_code}: {r.text[:200]}")
    data = r.json()["data"]
    return [d["embedding"] for d in data]


def _embed_gemini(texts, is_query):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    task = _GEMINI_QUERY_TASK if is_query else _GEMINI_DOC_TASK
    resp = client.models.embed_content(
        model=GEMINI_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=GEMINI_DIM),
    )
    return [e.values for e in resp.embeddings]


def _embed_openai(texts, is_query):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed(texts, is_query=False, api_key=None):
    """Embed a list of strings -> L2-normalized float32 array (n, dim).

    api_key overrides the env key for that call (used for multi-key rotation).
    """
    if isinstance(texts, str):
        texts = [texts]
    p = provider()
    if p == "cohere":
        vecs = _embed_cohere(texts, is_query, api_key=api_key)
    elif p == "voyage":
        vecs = _embed_voyage(texts, is_query)
    elif p == "gemini":
        vecs = _embed_gemini(texts, is_query)
    elif p == "openai":
        vecs = _embed_openai(texts, is_query)
    else:
        raise RuntimeError(f"Unknown EMBED_PROVIDER: {p}")
    return _normalize(vecs)


def embed_query(text):
    return embed([text], is_query=True)[0]
