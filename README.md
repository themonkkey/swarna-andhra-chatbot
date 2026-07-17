# Swarna Andhra GVA Assistant (prototype)

Chatbot over PIF's Swarna Andhra capacity-building training material. Scope:
1. GDP/GSDP estimation methodologies (3 types), bottom-up vs top-down, which sector uses which
2. GVA calculation method
3. District economic profile snapshots — comparative-advantage sectors + GVA-boosting interventions
4. GSDP/DDP district data — **loaded**: 28 AP districts × ~24 sectors × 4 years, parsed from the
   district GVA workbook (see "District dataset" below) into precise per-district-sector chunks
   plus a comparative-advantage snapshot per district.

## Stack (chosen for lowest cost)
- **Retrieval**: TF-IDF over the training corpus (scikit-learn) — no embedding API cost, runs instantly on ~1,300 chunks.
- **Generation**: Groq's free-tier API (Llama 3.1 8B) by default. Gemini free tier supported as an alternative.
- **UI**: Streamlit — free to run locally, free to host on Streamlit Community Cloud.

### Known limitation: TF-IDF is keyword matching, not semantic
It matches literal words/n-grams, so a query like "fisheries case study" won't reliably surface
the Srikakulam/Nellore/Biofloc case studies if their exact phrasing differs from the query
(e.g. "shrimp aquaculture" vs "fisheries"). We mitigated this two ways:
- Stripped repeating deck boilerplate ("CASE STUDY | ...", "Editable PowerPoint | N slides") that
  was spamming TF-IDF term frequency and drowning out real content (`BOILERPLATE_PATTERNS` in `ingest.py`).
- Added a manual sector-synonym tag (`CASE_STUDY_TAGS` in `ingest.py`) to each case study so common
  synonyms are still matchable.
It's still not semantic search. If retrieval quality on sector-based case-study queries matters
more than the cost of a real embedding model, the clean upgrade path is: swap `TfidfVectorizer` +
`cosine_similarity` in `ingest.py`/`app.py` for sentence-transformers (or a hosted embeddings API)
— the rest of the pipeline (chunking, chat loop, citations) doesn't need to change.

## Setup
```bash
cd swarna-andhra-chatbot
source venv/bin/activate   # already created
```

1. Put the source files (the 44 files from the shared Drive folder) in `corpus_files/`
   (already pointed at `~/Downloads/PIF_Shared_Folder` — see `ingest.py`'s `CORPUS_DIR`,
   or copy them into `corpus_files/` directly).
2. Build the index:
   ```bash
   python ingest.py
   ```
3. Get a free API key:
   - Groq (default): https://console.groq.com/keys — free tier, fast Llama models.
   - Or Gemini: https://aistudio.google.com/apikey — free tier.
4. Run:
   ```bash
   export GROQ_API_KEY=your_key_here
   streamlit run app.py
   ```
   (or `export LLM_PROVIDER=gemini` and `export GEMINI_API_KEY=...` to use Gemini instead)

## District dataset (item 4)
Source: `~/Downloads/1 DB 08-03-2026.xlsx` — district-wise GVA/GSDP/GDDP/NDDP by sector, 4 years
(2022-23 TRE, 2023-24 SRE, 2024-25 FRE, 2025-26 FAE), with Value/Rank/Growth%/Contribution% per
year, for all 28 AP districts.

Parsed by `parse_district_data.py` into:
- `structured_district_data.csv` — full long-format table (district, sector, year, value, rank,
  growth%, contribution%) — the exact-number source of truth if you need to compute/verify anything
  outside the chatbot.
- `corpus_files/District_Data/<District>/<Sector>.txt` — one small chunk per district-sector with
  all 4 years' figures in plain prose (precise, retrievable by exact district+sector queries).
- `corpus_files/District_Data/<District>_Snapshot.txt` — per-district summary: top sectors by
  contribution (comparative advantage), fastest-growing sectors, best statewide ranks, and
  aggregate GDDP/NDDP/population/per-capita figures (answers item 3's "profile snapshot" ask).

Re-run `python parse_district_data.py` if the workbook is updated, then `python ingest.py` to
rebuild the index.

## Deploying for free
Push this repo to GitHub, then deploy on https://streamlit.io/cloud (free tier) — set
`GROQ_API_KEY` (and `LLM_PROVIDER` if using Gemini) as a "secret" in the Streamlit Cloud app
settings instead of a local env var.
