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
import re

import numpy as np
import streamlit as st

try:
    from sklearn.metrics.pairwise import cosine_similarity  # only needed for legacy TF-IDF index
except Exception:
    cosine_similarity = None


def _bootstrap_secrets():
    """Make keys available as env vars whether local (.env) or Streamlit Cloud (st.secrets),
    so embeddings.py and the LLM call read them uniformly from os.environ."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    try:
        for k, v in st.secrets.items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass


_bootstrap_secrets()

_BASE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(_BASE, "index.pkl")
EMBED_NPZ = os.path.join(_BASE, "embed_index.npz")
EMBED_CHUNKS = os.path.join(_BASE, "embed_chunks.pkl")
# the 112MB matrix is committed as two <100MB parts (GitHub file-size limit) and
# reassembled here at load time
EMBED_PARTS = [os.path.join(_BASE, f"embed_index_part{i}.npz") for i in range(2)]


def _load_matrix():
    if os.path.exists(EMBED_NPZ):
        return np.load(EMBED_NPZ)["matrix"]
    if all(os.path.exists(p) for p in EMBED_PARTS):
        return np.concatenate([np.load(p)["matrix"] for p in EMBED_PARTS], axis=0)
    return None

# Recognizable district-name aliases for direct lookup — TF-IDF alone under-ranks a district
# snapshot against generic methodology docs when the query only has one distinctive term
# (e.g. "gdp of kakinada" loses to docs repeating "GDP"/"district" many times).
DISTRICT_ALIASES = {
    "alluri seetha rama raju": "Alluri_Seetha_Rama_Raju", "asr": "Alluri_Seetha_Rama_Raju",
    "anakapalle": "Anakapalle", "anakapalli": "Anakapalle",
    "ananthapuramu": "Ananthapuramu", "anantapur": "Ananthapuramu",
    "annamayya": "Annamayya",
    "bapatla": "Bapatla",
    "chittoor": "Chittoor",
    "konaseema": "Dr.B.R.Ambedkar_Konaseema", "ambedkar konaseema": "Dr.B.R.Ambedkar_Konaseema",
    "east godavari": "East_Godavari", "kakinada": "Kakinada",
    "eluru": "Eluru",
    "guntur": "Guntur",
    "krishna": "Krishna",
    "kurnool": "Kurnool",
    "markapuram": "Markapuram",
    "nandyal": "Nandyal",
    "ntr": "Ntr",
    "palnadu": "Palnadu",
    "parvathipuram manyam": "Parvathipuram_Manyam", "parvathipuram": "Parvathipuram_Manyam",
    "polavaram": "Polavaram",
    "prakasam": "Prakasam",
    "nellore": "Sps_Nellore", "sps nellore": "Sps_Nellore",
    "sri satya sai": "Sri_Satya_Sai", "satya sai": "Sri_Satya_Sai",
    "srikakulam": "Srikakulam",
    "tirupati": "Tirupati",
    "visakhapatnam": "Visakhapatnam", "vizag": "Visakhapatnam",
    "vizianagaram": "Vizianagaram",
    "west godavari": "West_Godavari",
    "ysr kadapa": "Ysr_Kadapa", "kadapa": "Ysr_Kadapa",
}


def detect_district(query):
    low = query.lower()
    for alias, folder in DISTRICT_ALIASES.items():
        if alias in low:
            return folder
    return None

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
    # semantic (embedding) index — preferred
    matrix = _load_matrix()
    if matrix is not None and os.path.exists(EMBED_CHUNKS):
        with open(EMBED_CHUNKS, "rb") as f:
            meta = pickle.load(f)
        return {"mode": "embed", "chunks": meta["chunks"],
                "matrix": matrix.astype(np.float32), "model_id": meta.get("model_id")}
    # legacy TF-IDF fallback
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)
        idx["mode"] = "tfidf"
        return idx
    return None


def _label(source, page):
    return f"{source} (slide/page {page})" if page else source


def _scores(query, index):
    """Similarity of query against every chunk, for whichever index mode is loaded."""
    if index["mode"] == "embed":
        import embeddings
        qvec = embeddings.embed_query(query)  # L2-normalized
        return index["matrix"] @ qvec  # cosine == dot product
    qvec = index["vectorizer"].transform([query])
    return cosine_similarity(qvec, index["matrix"]).flatten()


def retrieve(query, index, district_folder=None):
    if index is None:
        return []
    sims = _scores(query, index)
    top_score = float(sims.max())
    # embedding cosines sit higher than TF-IDF; thresholds tuned per mode
    if index["mode"] == "embed":
        k = 6 if top_score > 0.75 else 10 if top_score > 0.6 else 16
    else:
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

    # a detected district name is a stronger signal than TF-IDF score — force its snapshot
    # and sector files in first so they aren't drowned out by generic methodology docs.
    if district_folder:
        sector_prefix = f"district_data/{district_folder}/"
        snapshot_name = f"district_data/{district_folder}_Snapshot.txt"
        forced = [
            (j, c) for j, c in enumerate(index["chunks"])
            if c["source"].startswith(sector_prefix) or c["source"] == snapshot_name
        ]

        # headline aggregates (GDDP, NDDP, per-capita, GDVA) and the snapshot must come
        # before the alphabetical per-sector files, or the truncated context drops them.
        def prio(src):
            s = src.lower()
            if "snapshot" in s:
                return 0
            if "gross district domestic product" in s or "gddp" in s:
                return 1
            if "net district domestic product" in s or "nddp" in s:
                return 2
            if "per capita" in s:
                return 3
            if "gross district value added" in s or "gdva" in s:
                return 4
            return 5

        # force only the top few headline aggregates (snapshot carries GDDP/NDDP/
        # per-capita/top-sectors already). Forcing all ~7 buried the semantically
        # relevant mandal/constituency vision plans for topical queries like
        # "paddy productivity in Anaparthi mandal", so keep this lean.
        forced.sort(key=lambda jc: prio(jc[1]["source"]))
        for j, c in forced[:2]:
            add_chunk(j, 1.0)

    primary = [i for i in top_idx if sims[i] > 0]

    # keyword rescue: numeric tables (e.g. "Paddy Productivity 7275 -> 7646") embed
    # poorly, so pure vector search finds the right DOCUMENT but often the wrong PAGE.
    # For each top-matching document, also pull the page whose text best matches the
    # query's content words. Added before the primary hits so it survives the cap.
    stop = {"what", "is", "the", "and", "its", "for", "of", "in", "how", "which",
            "are", "to", "me", "tell", "about", "give", "show", "district", "mandal",
            "constituency", "target", "targets", "plan"}
    qwords = set(w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 3 and w not in stop)
    if qwords:
        top_sources = []
        for i in primary[:5]:
            s = index["chunks"][i]["source"]
            if s not in top_sources:
                top_sources.append(s)
        # queries seeking figures ("productivity", "target", "growth"...) want the
        # number-bearing table page, which embeds poorly and has few keywords
        data_intent = bool(re.search(
            r"productiv|target|growth|\brate\b|income|gdp|gddp|gsdp|\bddp\b|per capita|"
            r"population|contribution|hectare|\barea\b|yield|percent|\bvalue\b|figure",
            query.lower()))
        for src in top_sources[:3]:
            # exclude query words that are just the place/name (they appear in the
            # document's own path), so within-document ranking uses topic words only
            path_words = set(re.findall(r"[a-z]+", src.lower()))
            topic = qwords - path_words
            if not topic:
                continue
            scored = []
            for j, c in enumerate(index["chunks"]):
                if c["source"] != src:
                    continue
                low = c["text"].lower()
                cover = sum(1 for w in topic if w in low)
                if cover == 0:
                    continue
                digits = len(re.findall(r"\d", c["text"])) if data_intent else 0
                scored.append((cover * 100 + min(digits, 60), j))
            scored.sort(reverse=True)
            for _, j in scored[:2]:
                add_chunk(j, 0.99)

    # pass 1: all distinct primary hits, so no single document's neighbor pages
    # crowd a more relevant document out of the context window
    for i in primary:
        add_chunk(i, float(sims[i]))

    # pass 2: neighbor expansion (adjacent pages) for multi-page PDFs, appended after
    for i in primary:
        c = index["chunks"][i]
        if c.get("page") is not None and c.get("folder") in ("methodology", "vision_documents"):
            for delta in (-1, +1):
                neighbor_key = (c["source"], c["page"] + delta)
                if neighbor_key in page_index:
                    add_chunk(page_index[neighbor_key], float(sims[i]) * 0.9, neighbor=True)

    return results


# Groq's free tier caps llama models at a few thousand tokens/minute, so the prompt
# must stay small. Cap how many chunks and how much of each go into the LLM context
# (the full hit list is still shown separately under "sources retrieved").
CONTEXT_MAX_CHUNKS = 11
CONTEXT_CHARS_PROSE = 900       # vision/methodology prose — trim hard
CONTEXT_CHARS_DATA = 1600       # district_data files are short + dense with exact numbers


def build_context_block(hits):
    if not hits:
        return "(No relevant material found in the PIF corpus for this query.)"
    parts = []
    for h in hits[:CONTEXT_MAX_CHUNKS]:
        label = _label(h["source"], h["page"])
        cap = CONTEXT_CHARS_DATA if h["source"].startswith("district_data/") else CONTEXT_CHARS_PROSE
        text = h["text"][:cap]
        parts.append(f"--- Source: {label} (relevance {h['score']:.2f}) ---\n{text}")
    return "\n\n".join(parts)


def call_llm(messages):
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            messages=messages,
            temperature=0.2,
            max_tokens=1024,
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


st.set_page_config(
    page_title="Swarna Andhra GVA Assistant",
    page_icon="🏛️",
    layout="centered",
)

BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');
:root{--bg:#17201C;--bg-2:#1E2A24;--panel:#212E27;--neon:#01EF6C;--neon-dk:#00C957;--line:#2E3E35;--ink:#EAF3ED;--muted:#8FA598;}
html,body,[class*="css"],.stApp{font-family:'Inter',sans-serif;background:var(--bg) !important;color:var(--ink);}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;}
.block-container{padding-top:1.2rem;padding-bottom:7rem;max-width:820px;}
.stApp{background:var(--bg);}

/* branded header */
.sa-header{display:flex;align-items:center;gap:14px;background:var(--bg-2);border:1px solid var(--line);border-radius:14px;padding:20px 24px;margin-bottom:6px;position:relative;overflow:hidden;}
.sa-header::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--neon);}
.sa-emblem{width:44px;height:44px;border-radius:10px;background:rgba(1,239,108,.10);border:1px solid rgba(1,239,108,.30);display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0;}
.sa-htext h1{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:24px;color:var(--ink);line-height:1.05;margin:0;letter-spacing:-.01em;}
.sa-htext h1 span{color:var(--neon);}
.sa-htext p{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);margin:5px 0 0;font-weight:400;letter-spacing:.02em;}
.sa-badge{margin-left:auto;background:rgba(1,239,108,.12);color:var(--neon);border:1px solid rgba(1,239,108,.35);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;padding:5px 12px;border-radius:6px;white-space:nowrap;text-transform:uppercase;letter-spacing:.08em;}
.sa-sub{font-size:13px;color:var(--muted);margin:12px 2px 4px;font-weight:300;line-height:1.65;}

/* chat bubbles */
[data-testid="stChatMessage"]{background:transparent;padding:.2rem 0;}
[data-testid="stChatMessageContent"]{font-size:15px;line-height:1.65;color:var(--ink);}
/* assistant message card */
.stChatMessage:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"]{
  background:var(--panel);border:1px solid var(--line);border-radius:4px 14px 14px 14px;padding:14px 18px;border-left:3px solid var(--neon);}
/* user message card */
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]{
  background:var(--neon);color:#06130C;border-radius:14px 4px 14px 14px;padding:12px 18px;font-weight:500;}
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] *{color:#06130C !important;}

/* welcome chips */
div.stButton>button{background:var(--bg-2);border:1px solid var(--line);color:var(--ink);border-radius:10px;font-size:13px;
  font-weight:400;padding:11px 16px;text-align:left;transition:all .12s;font-family:'Inter';}
div.stButton>button:hover{border-color:var(--neon);background:var(--panel);color:var(--neon);}

/* sources expander */
[data-testid="stExpander"]{border:none;background:transparent;}
[data-testid="stExpander"] summary{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--muted);font-weight:400;text-transform:uppercase;letter-spacing:.06em;}
[data-testid="stExpander"] summary:hover{color:var(--neon);}
[data-testid="stExpander"] [data-testid="stExpanderDetails"]{color:var(--muted);font-size:12px;}

/* chat input */
[data-testid="stChatInput"]{border:1px solid var(--line);border-radius:12px;background:var(--bg-2);}
[data-testid="stChatInput"]:focus-within{border-color:var(--neon);box-shadow:0 0 0 1px rgba(1,239,108,.25);}
[data-testid="stChatInput"] textarea{font-size:15px;color:var(--ink);}
[data-testid="stChatInput"] button{background:var(--neon);border-radius:9px;}
[data-testid="stChatInput"] button svg{color:#06130C;fill:#06130C;}

a{color:var(--neon) !important;}
.sa-foot{text-align:center;font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);margin-top:8px;letter-spacing:.04em;}
</style>
"""
st.markdown(BRAND_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="sa-header">
      <div class="sa-emblem">🏛️</div>
      <div class="sa-htext">
        <h1>Swarna Andhra <span>GVA Assistant</span></h1>
        <p>Pahlé India Foundation · aligned with Swarna Andhra @2047</p>
      </div>
      <div class="sa-badge">Prototype</div>
    </div>
    <div class="sa-sub">Ask about GDP / GSDP estimation, GVA calculation, district economic profiles,
    or any constituency and mandal vision plan — answered from official material with sources.</div>
    """,
    unsafe_allow_html=True,
)

index = load_index()
if index is None:
    st.error("No index found. Run `python embed_index.py` to build the corpus index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

EXAMPLES = [
    "What is the GDDP and per capita income of Kakinada?",
    "How is district income estimated — top-down or bottom-up?",
    "Economic priorities in the Bapatla constituency vision plan?",
    "Which sectors give Visakhapatnam its comparative advantage?",
]

# welcome screen with starter questions (only before the first message)
if not st.session_state.messages:
    st.markdown("<div class='sa-sub' style='margin-top:18px;font-weight:500;color:var(--ink)'>Try asking</div>",
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

# replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])


def handle_query(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    district_folder = detect_district(user_input)
    # follow-up handling: carry over the last district mentioned when this turn names none
    if not district_folder:
        for m in reversed(st.session_state.messages[:-1]):
            if m["role"] == "user":
                prev = detect_district(m["content"])
                if prev:
                    district_folder = prev
                    break

    hits = retrieve(user_input, index, district_folder=district_folder)
    context_block = build_context_block(hits)
    history = [
        {"role": m["role"], "content": m["content"][:600]}
        for m in st.session_state.messages[:-1][-4:]
    ]
    llm_messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"CONTEXT:\n{context_block}\n\nQUESTION: {user_input}"}]
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus…"):
            try:
                answer = call_llm(llm_messages)
            except KeyError as e:
                answer = f"Missing API key: {e}."
            except Exception as e:
                answer = f"Sorry, something went wrong: {e}"
        st.markdown(answer)
        sources_md = ""
        if hits:
            seen = []
            for h in hits:
                lbl = _label(h["source"], h["page"])
                if lbl not in seen:
                    seen.append(lbl)
            sources_md = "\n".join(f"- `{s}`" for s in seen[:8])
            with st.expander("Sources"):
                st.markdown(sources_md)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_md}
    )


typed = st.chat_input("Ask about a district, a vision plan, or GVA methodology…")
query = typed or st.session_state.pending
st.session_state.pending = None
if query:
    handle_query(query)
    st.rerun()

st.markdown("<div class='sa-foot'>Grounded in official Swarna Andhra training material · answers may be approximate</div>",
            unsafe_allow_html=True)
