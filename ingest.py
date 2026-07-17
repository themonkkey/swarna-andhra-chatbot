"""Extract text from the PIF training corpus with page/slide-level indexing.

Each chunk = one slide (pptx) or one page (pdf), so retrieval can cite an exact
location instead of an arbitrary character window. Long pages are sub-split
(with overlap) but keep their page number, so citation stays precise.
Docx templates have no native page concept — indexed as a single unit per file.
"""
import os
import pickle
import re

import docx
import pdfplumber
from pptx import Presentation
from sklearn.feature_extraction.text import TfidfVectorizer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus_files")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")

# TF-IDF only matches literal tokens, so it misses queries phrased with synonyms the
# source text doesn't use (e.g. "fisheries" when the deck says "shrimp"/"aquaculture").
# Tag known files with their sector synonyms so every chunk of that file carries the
# extra tokens needed for recall, without pulling in a full embedding model.
CASE_STUDY_TAGS = {
    "Srikakulam_Blue_Economy.txt": "fisheries aquaculture shrimp fish marine coastal blue economy",
    "Nellore_Shrimp_Processing.txt": "fisheries aquaculture shrimp seafood processing export",
    "Biofloc_Tilapia_CaseStudy.txt": "fisheries aquaculture fish tilapia biofloc pond farming",
    "Paddy_Fish_Integrated_Farming_Case_Study_AP.txt": "fisheries aquaculture fish paddy integrated farming",
    "East_Godavari_Coconut_Coir.txt": "coconut coir horticulture agro processing waste to wealth",
    "Nellore_Ethanol_Potential.txt": "ethanol biofuel sugarcane maize distillery energy manufacturing",
    "Banana_Processing_Case_Study.txt": "banana horticulture agro processing waste to wealth pseudo-stem",
    "Kumarakom_Responsible_Tourism_Case_Study.txt": "tourism hospitality services responsible tourism ecotourism",
    "Shenzhen_Port_Led_Manufacturing.txt": "manufacturing industry port logistics special economic zone china",
    "Sahyadri_Replication_Playbook.txt": "grapes horticulture fpo fpc farmer producer company cold chain",
    "Sahyadri_V2.txt": "chilli spices agriculture fpo fpc value chain export prakasam",
    "Chetna_FPO_Lessons.txt": "organic cotton textiles fpo shg cooperative farmer producer company",
    "Morbi_Ceramics_Industry.txt": "ceramics tiles manufacturing industry cluster gujarat",
    "Tiruppur_Case_Study_Updated.txt": "textiles knitwear garments manufacturing export cluster zero liquid discharge",
}

# Only pages longer than this get sub-split (with overlap); most slides/pages fit in one chunk.
MAX_PAGE_CHARS = 1200
SUB_CHUNK_OVERLAP = 150


def extract_docx_units(path):
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [{"page": None, "text": "\n".join(parts)}]


def extract_pptx_units(path):
    prs = Presentation(path)
    units = []
    for i, slide in enumerate(prs.slides, 1):
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(r.text for r in para.runs)
                    if t.strip():
                        slide_text.append(t)
            if shape.has_table:
                for row in shape.table.rows:
                    slide_text.append(" | ".join(c.text.strip() for c in row.cells))
        if slide_text:
            units.append({"page": i, "text": "\n".join(slide_text)})
    return units


def extract_pdf_units(path):
    units = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            t = page.extract_text() or ""
            if t.strip():
                units.append({"page": i, "text": t})
    return units


def extract_txt_units(path):
    # Fallback for files whose original binary wouldn't parse — no native page
    # structure preserved, so indexed as a single unit.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [{"page": None, "text": f.read()}]


def extract_units(path):
    ext = path.lower().rsplit(".", 1)[-1]
    try:
        if ext == "docx":
            return extract_docx_units(path)
        if ext == "pptx":
            return extract_pptx_units(path)
        if ext == "pdf":
            return extract_pdf_units(path)
        if ext == "txt":
            return extract_txt_units(path)
    except Exception as e:
        print(f"  ! failed to extract {path}: {e}")
    return []


# Recurring deck boilerplate that repeats on nearly every slide of the case-study decks —
# left in, it spams the "case study" bigram and drowns out the actual topical content in
# TF-IDF scoring (a query like "fisheries case study" then matches whichever deck repeats
# this boilerplate most, not whichever deck is actually about fisheries).
BOILERPLATE_PATTERNS = [
    re.compile(r"^CASE STUDY\s*\|.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r".*Training deck for District.*Mandal Level Officials.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Editable PowerPoint\s*\|.*slides.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^Source:.*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r".*\|\s*Case study for AP officials.*$", re.IGNORECASE | re.MULTILINE),
]


def strip_boilerplate(text):
    for pat in BOILERPLATE_PATTERNS:
        text = pat.sub("", text)
    return text


def sub_split(text, page, source):
    """Split an overlong page/slide at paragraph boundaries, keeping the same page label."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= MAX_PAGE_CHARS:
        return [{"source": source, "page": page, "text": text}]
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > MAX_PAGE_CHARS and current:
            chunks.append({"source": source, "page": page, "text": current.strip()})
            # carry overlap from end of previous chunk into next
            current = current[-SUB_CHUNK_OVERLAP:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip()
    if current.strip():
        chunks.append({"source": source, "page": page, "text": current.strip()})
    return chunks


def main():
    all_chunks = []
    for root, _, files in os.walk(CORPUS_DIR):
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = fname.lower().rsplit(".", 1)[-1]
            if ext not in ("docx", "pptx", "pdf", "txt"):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, CORPUS_DIR)
            print(f"Extracting: {rel}")
            units = extract_units(path)
            file_chunks = []
            tags = CASE_STUDY_TAGS.get(fname)
            is_case_study = "Some_case_studies" in rel or "Some case studies" in rel
            for u in units:
                unit_text = strip_boilerplate(u["text"]) if is_case_study else u["text"]
                for chunk in sub_split(unit_text, u["page"], rel):
                    if tags:
                        # Repeat so tag terms carry enough TF-IDF weight to compete with
                        # the surrounding jargon-heavy slide text, not just appear once.
                        tag_block = " ".join([tags] * 4)
                        chunk["text"] = f"Keywords: {tag_block}\n\n{chunk['text']}"
                    file_chunks.append(chunk)
            print(f"  -> {len(units)} pages/slides -> {len(file_chunks)} chunks")
            all_chunks.extend(file_chunks)

    if not all_chunks:
        print("No chunks extracted — check CORPUS_DIR path and file downloads.")
        return

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "vectorizer": vectorizer, "matrix": matrix}, f)

    print(f"\nIndexed {len(all_chunks)} chunks (page/slide-level) from corpus into {INDEX_PATH}")


if __name__ == "__main__":
    main()
