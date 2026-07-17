"""Swarna Andhra training-material chatbot — Streamlit prototype.

Scope (per Aryan's brief):
1. GDP/GSDP estimation methodologies (3 types), bottom-up vs top-down, which sector uses which
2. GVA calculation method
3. District economic profile snapshots — comparative-advantage sectors + GVA-boosting interventions
4. GSDP/DDP data of districts (pending a structured dataset — flagged when unavailable)

Retrieval is grounded in the PIF training corpus (TF-IDF over training decks + case studies).
Falls back to the model's general knowledge when the corpus doesn't cover a question,
but is told to say so explicitly rather than blur the two.
"""
import os
import pickle

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")

SYSTEM_PROMPT = """You are an assistant for Pahlé India Foundation's Swarna Andhra capacity-building \
programme, which trains Andhra Pradesh district/constituency/mandal officials on GDP/GSDP/GDDP \
estimation and sector-wise GVA improvement.

You are focused on exactly four topics:
1. GDP/GSDP estimation methodologies (production, income, expenditure approaches), bottom-up vs \
top-down approaches, and which sector typically uses which method.
2. GVA calculation methodology.
3. Economic profile snapshots of AP districts — sectors with comparative advantage, and possible \
interventions/suggestions to boost GVA in those sectors.
4. GSDP/DDP data of specific districts (numeric figures), when available in context.

You are given CONTEXT chunks retrieved from PIF's own training material (decks, case studies, \
toolkits), each labeled with its source file and, where available, a slide/page number. \
Ground your answer in that context first and cite which source file (and slide/page, if given) \
you drew from. \
If the context doesn't fully answer the question, you may supplement with general economic \
knowledge — but say explicitly which part of your answer is from the PIF corpus and which part is \
general knowledge/not verified against official data.

If asked for district-level GSDP/DDP numeric data that is not in the context, say plainly that you \
don't have that specific figure rather than guessing numbers.
"""


@st.cache_resource
def load_index():
    if not os.path.exists(INDEX_PATH):
        return None
    with open(INDEX_PATH, "rb") as f:
        return pickle.load(f)


def _label(source, page):
    return f"{source} (slide/page {page})" if page else source


def retrieve(query, index):
    if index is None:
        return []
    qvec = index["vectorizer"].transform([query])
    sims = cosine_similarity(qvec, index["matrix"]).flatten()
    top_score = float(sims.max())
    k = 5 if top_score > 0.3 else 9 if top_score > 0.15 else 14
    top_idx = sims.argsort()[::-1][:k]

    seen_keys = set()
    results = []

    def add_chunk(i, score, neighbor=False):
        c = index["chunks"][i]
        key = (c["source"], c.get("page"))
        if key in seen_keys:
            return
        seen_keys.add(key)
        results.append({
            "source": c["source"],
            "page": c.get("page"),
            "text": c["text"],
            "score": score,
            "neighbor": neighbor,
        })

    # build a lookup: (source, page) -> chunk index, for neighbor expansion
    page_index = {}
    for j, c in enumerate(index["chunks"]):
        if c.get("page") is not None:
            page_index[(c["source"], c["page"])] = j

    for i in top_idx:
        if sims[i] <= 0:
            continue
        add_chunk(i, float(sims[i]))
        c = index["chunks"][i]
        # neighbor expansion only for methodology PDFs — training/case_study slides are self-contained
        if c.get("page") is not None and c.get("folder") == "methodology":
            for delta in (-1, +1):
                neighbor_key = (c["source"], c["page"] + delta)
                if neighbor_key in page_index:
                    add_chunk(page_index[neighbor_key], float(sims[i]) * 0.9, neighbor=True)

    return results


def build_context_block(hits):
    if not hits:
        return "(No relevant material found in the PIF corpus for this query.)"
    parts = []
    for h in hits:
        label = _label(h["source"], h["page"])
        parts.append(f"--- Source: {label} (relevance {h['score']:.2f}) ---\n{h['text']}")
    return "\n\n".join(parts)


def call_llm(messages):
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content
    elif provider == "gemini":
        from google import genai

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        prompt = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        resp = client.models.generate_content(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"), contents=prompt
        )
        return resp.text
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")


st.set_page_config(page_title="Swarna Andhra GVA Assistant", page_icon="📊")
st.title("📊 Swarna Andhra GVA Assistant (prototype)")
st.caption(
    "Ask about GDP/GSDP estimation methods, GVA calculation, or district economic profiles & "
    "GVA-boosting interventions — grounded in PIF's training corpus."
)

index = load_index()
if index is None:
    st.error(
        "No index found. Run `python ingest.py` first to build the corpus index from "
        "`corpus_files/`."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("e.g. How is fisheries GVA estimated — top-down or bottom-up?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    expanded = user_input
    low = user_input.lower()
    if any(w in low for w in ["gdp", "gddp", "economy", "economic", "income", "nddp", "per capita"]):
        expanded = user_input + " GDDP district domestic product snapshot"
    hits = retrieve(expanded, index)
    context_block = build_context_block(hits)

    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {user_input}",
        },
    ]

    with st.chat_message("assistant"):
        try:
            answer = call_llm(llm_messages)
        except KeyError as e:
            answer = (
                f"Missing API key: {e}. Set it as an environment variable before running "
                "`streamlit run app.py` (see README)."
            )
        except Exception as e:
            answer = f"Error calling the LLM: {e}"
        st.markdown(answer)
        if hits:
            with st.expander("Sources retrieved from corpus"):
                for h in hits:
                    tag = " *(neighbor)*" if h.get("neighbor") else ""
                    st.markdown(f"- `{_label(h['source'], h['page'])}` (score {h['score']:.2f}){tag}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
